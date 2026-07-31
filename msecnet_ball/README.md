# MSECNet Center-Ball Normal Estimation

This is an isolated experiment for estimating the camera-oriented normal at a
reviewed human 3D center from an 8 cm radius point-cloud ball.  It does not
modify or share checkpoints with `msecnet/`.

For every sample, preparation selects all source points satisfying
`||point - center_3d|| <= 0.08 m`.  Training then uses:

- coordinates: `(point - center_3d) / 0.08`, so the target location is always
  `(0, 0, 0)` and a fixed physical scale is retained;
- one point feature: normalized distance `r = ||coordinate||`, clamped to
  `[0, 1]` after training jitter;
- radial weighting: each point contributes `exp(-beta * r^2)` to both point
  loss and final vector pooling.  `beta=2` gives a boundary point weight of
  about 0.135 relative to a center point.  Set `--radial-weight-beta 0` for an
  ablation without distance-based weighting while retaining the distance input.

The target remains the reviewed, camera-oriented normal at `center_3d`; points
from nearby walls, rims, and holes are input context, not separate targets.

## Prepare the 8 cm dataset

Run from the project root in the required Conda environment:

```bash
cd /data2/shendu/code/ruoyu/train_point2normal

conda run --no-capture-output -n point2normal python \
  msecnet_ball/prepare_ball_dataset.py \
  data/fuelcap_pass_20260721_9211 \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08 \
  --ball-radius-m 0.08 \
  --max-points 4096
```

The preparation output retains the existing group-disjoint train/validation/test
split, but all `clouds/*.npz` files are newly produced spherical patches.

## Train

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
  --ball-radius-m 0.08 \
  --radial-weight-beta 2.0 \
  --steps 70000 \
  --bs 24 \
  --max-points 1024 \
  --aug-deg 45 \
  --point-loss-weight 0.25 \
  --seed 20260722 \
  --early-stop-patience 100 \
  --snapshot-every 1000 \
  --out msecnet_ball/out/center_ball_r08_oriented_9211_v1
```

The extra one-dimensional input makes the model weights incompatible with the
zero-feature `msecnet/` checkpoints, so this experiment must train from
scratch.

## Evaluate

```bash
conda run --no-capture-output -n point2normal python msecnet_ball/infer.py \
  msecnet_ball/out/center_ball_r08_oriented_9211_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
  --split-name test
```

`infer.py` obtains the radius and beta from checkpoint metadata and rejects
non-ball checkpoints.  Use the same `beta` for comparison experiments unless
the ablation is intentional.

## Web Viewer

After test inference, show the read-only report in the browser. The viewer
loads the prepared ball patch, original source context, target normal, and the
ball-model prediction.

```bash
conda run --no-capture-output -n point2normal python web_label/server.py \
  --msecnet-report msecnet_ball/out/center_ball_r08_oriented_9211_v1/inference_test/report.json \
  --msecnet-dataset data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08 \
  --msecnet-source-root data/fuelcap_pass_20260721_9211 \
  --port 8766
```

Open `http://localhost:8766`. To choose pseudo-OBB or 8 cm ball checkpoints
and launch matching inference from the browser, start the server with
`--msecnet-ui` instead of `--msecnet-report`.

## Unlabeled OBB Crops

For unlabeled detector-OBB datasets, the ball predictor uses the OBB crop
centroid only as a proxy 3D center. It then selects the 8 cm ball from the
uncropped `source_cloud` PLY, matching the training geometry. This is an
inference-time proxy, not a replacement for reviewed training centers.

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/predict_unlabeled.py \
  msecnet_ball/out/center_ball_r08_oriented_9211_v1/best.pt \
  data/msecnet_20260730_open_obb10_unlabeled_test \
  --source-root raw_data/20260730_五辆车外盖_多角度_三分类/open
```

The report contains predictions and aggregation strength only; no angular
error is available because this dataset has no target normals.

## Visualize Random Balls

Export ten deterministic random PLYs with gray source context, blue 8 cm ball
points, and the yellow human target normal:

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball/export_pseudo_obb_ply.py \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08 \
  --source-root data/fuelcap_pass_20260721_9211 \
  --random-count 10 \
  --seed 20260723 \
  --raw-points 50000
```

Files are written to `DATASET/visualizations/random_ball_seed_20260723/` with
an `index.json` that maps every PLY to its dataset sample.  Use `--split val`
or `--split test` to sample from a specific split.

## CPU checks

```bash
conda run --no-capture-output -n point2normal python -m unittest \
  msecnet_ball.test_ball_objective
```
