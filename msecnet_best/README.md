# MSECNet Best Historical Run

This directory preserves the code and checkpoint for the historical
`ckpt_msecnet_v4_manual_pseudo_obb` run. It is based on Git commit
`f56b018` (`训练回归效果误差3度`), which is the version that produced the
included checkpoint.

The checkpoint was selected at step 53,900. Its validation metrics were:

| Metric | Value |
| --- | ---: |
| Mean axial angular error | 3.530 deg |
| Median axial angular error | 2.388 deg |
| Axial error <= 10 deg | 94.57% |

The held-out car-model split report contains 587 samples with 2.407 deg mean
axial angular error. The saved report and training metrics are under
`artifacts/`.

## Contents

| Path | Purpose |
| --- | --- |
| `train.py` | Historical sign-invariant training objective |
| `infer.py` | Historical sign-invariant split inference |
| `checkpoints/best.pt` | Checkpoint selected at step 53,900 |
| `MSECNet/` | MSECNet architecture and CUDA point operations |
| `cap_patch.py` | Historical local-patch dependency bundled for portability |

The only portability changes from commit `f56b018` are that `train.py` imports
the bundled `cap_patch.py`, and `infer.py` defaults to the bundled checkpoint.
The model architecture, dataset handling, aggregation, loss, and training
parameters are unchanged.

This is a legacy, sign-invariant model: normal direction is evaluated with
`abs(dot(prediction, target))`, so opposite vector directions are equivalent.
Do not compare its axial-angle metrics directly with later camera-oriented
models.

## Environment

Run every command from the project root in the `point2normal` Conda
environment:

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
```

The MSECNet `pointops` CUDA extension must already be compiled for this
environment.

## Test Inference

The default arguments reproduce the historical test-split setup. Results are
written to `msecnet_best/checkpoints/inference_test/` unless `--out` is given.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py
```

Equivalent explicit command:

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  msecnet_best/checkpoints/best.pt \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/split_by_car_model.json \
  --split-name test
```

## Train Again

This command uses the same intended training settings as the saved run. A
fresh run is not bit-for-bit deterministic because the historical script uses
time-based training sampling.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/train.py \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/split_by_car_model.json \
  --steps 70000 --bs 24 --max-points 1024 \
  --radius 0.3 --aug-deg 45 \
  --out msecnet_best/out/ckpt_msecnet_v4_manual_pseudo_obb_retrain
```

Important historical settings encoded in the checkpoint are `max_points=1024`,
`aug_deg=45`, variable-size point batches, and the loss
`1 - dot(normalize(point_prediction), target)^2`.

Each new training run writes these files to its `--out` directory:

| Path | Contents |
| --- | --- |
| `metrics.csv` | Fixed-column CSV for training loss, learning rate, and validation metrics |
| `dashboard.png` | Training loss, validation axial angle error, `<=10 deg` accuracy, and learning rate |
| `run.json` | Command-line arguments, sample counts, aggregation, and loss definition |
| `last.pt`, `best.pt` | Latest validation checkpoint and lowest-validation-error checkpoint |
| `snapshots/step_*.pt` | Periodic checkpoints controlled by `--snapshot-every` |
