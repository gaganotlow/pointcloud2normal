# PointNet++ Center-Ball Normal Regression

This is a pure-PyTorch PointNet++ baseline for the same 8 cm center-ball data
used by `msecnet_ball/`.  It regresses **one** camera-oriented unit normal from
each local point cloud; it does not assign the ball-center target to every
input point.

The repaired 5 cm derivative is now available at:

```text
data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r05/
```

It preserves the current repaired labels, anchors, and the exact r08
train/val/test split.  Its clouds are regenerated from the original full
clouds, not cropped from an already sampled 8 cm subset.

## Model and input contract

```text
fixed-size 8 cm ball [N, 3] + radial distance [N, 1]
  -> Set Abstraction 1: FPS + kNN + shared MLP
  -> Set Abstraction 2: FPS + kNN + shared MLP
  -> global max pooling
  -> MLP(3) -> L2 normalization
```

Coordinates are `(point - center_3d) / 0.08`.  Every batch has exactly
`--num-points` points: larger balls are sampled without replacement and sparse
balls repeat points.  Training rotation applies the same rotation to the
coordinates and target normal.  The directed objective is:

```text
1 - dot(normalize(prediction), normalize(target))
```

Thus a flipped normal is maximally wrong.  This directory uses standard
PyTorch tensor operations and does not need MSECNet's custom CUDA pointops;
CUDA is still recommended for normal training speed.

## Train

Run from the repository root in the required environment:

```bash
conda run --no-capture-output -n point2normal python \
  pointnet2_ball/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --ball-radius-m 0.08 --num-points 1024 --aug-deg 45 \
  --steps 70000 --batch-size 24 --lr 3e-4 \
  --ema-decay 0.995 --grad-clip 1.0 \
  --val-every 100 --early-stop-patience 100 --seed 20260722 \
  --out pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1
```

`best.pt`, `last.pt`, `metrics.csv`, and `run.json` are written to `--out`.
Model selection uses validation mean **oriented** angular error.  For a fair
comparison with MSECNet, retain the same data directory, group split, radius,
point count, rotation range, seed, and label revision.

This is the stabilized v2 recipe: a lower peak learning rate, global gradient
clipping, and EMA weights for validation/checkpoint selection.  After a first
run, generate a `train` report with `infer.py` and optionally emphasize its
confirmed hard samples in a second run:

```bash
在上一条训练命令的参数末尾追加：
  --init-checkpoint pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1/best.pt \
  --hard-report pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1/inference_train/report.json \
  --hard-threshold 10 --hard-weight 4 \
  --out pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v2_hard
```

The hard report must be regenerated after any label edits.  It changes sample
weights only; validation remains unweighted.

To train the 5 cm version, replace every `...r08...` dataset path in the
command above with `...r05...` and use `--ball-radius-m 0.05`.

To diagnose capacity versus label/input conflict, use the same report to select
the 32 worst training samples.  This mode automatically fixes point sampling,
disables rotation, jitter, EMA, and dropout, and evaluates on those same 32
samples:

```bash
conda run --no-capture-output -n point2normal python \
  pointnet2_ball/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --overfit-report pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1/inference_train/report.json \
  --overfit-count 32 --steps 5000 --batch-size 16 --num-points 1024 \
  --lr 3e-4 --workers 0 --val-workers 0 \
  --out pointnet2_ball/out/overfit_hard32
```

## Evaluate a split

```bash
conda run --no-capture-output -n point2normal python \
  pointnet2_ball/infer.py \
  pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --split-name test
```

The report format matches the existing ball inference reports, so the normal
repair Web app can queue `train`/`val` PointNet++ disagreements.  Its
`point_consensus` field is intentionally empty: unlike MSECNet, PointNet++ has
one global prediction head and no collection of per-point normal votes.

## Unlabeled OBB crops

The open-set command uses the same proxy-center protocol as `msecnet_ball`:
the OBB crop centroid selects an 8 cm ball from its original full PLY cloud.
It is only for prediction inspection, not model selection.

```bash
conda run --no-capture-output -n point2normal python \
  pointnet2_ball/predict_unlabeled.py \
  pointnet2_ball/out/pointnet2_ball_r08_oriented_20260803_v1/best.pt \
  data/msecnet_20260730_open_obb10_unlabeled_test \
  --source-root raw_data/20260730_五辆车外盖_多角度_三分类/open
```

## CPU checks

```bash
conda run --no-capture-output -n point2normal python -m unittest pointnet2_ball.test_pointnet2
```
