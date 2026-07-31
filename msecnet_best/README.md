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
| `export_onnx.py` | Export a fixed-size, standard-operator ONNX deployment model |
| `prepare_pseudo_obb_dataset.py` | Historical pseudo-OBB dataset preparation |
| `export_pseudo_obb_ply.py` | Export one prepared pseudo-OBB patch as a PLY |
| `checkpoints/best.pt` | Checkpoint selected at step 53,900 |
| `MSECNet/` | MSECNet architecture and CUDA point operations |
| `cap_patch.py` | Historical local-patch dependency bundled for portability |

The model architecture, dataset handling, aggregation, loss, and training
parameters match commit `f56b018`. The bundled version also fixes the training
CSV/dashboard recording and includes the historical dataset preparation and
PLY export utilities.

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

## Export ONNX

The original MSECNet uses CUDA-only point operations, which standard ONNX
cannot represent. `export_onnx.py` swaps those operations for exact PyTorch
KNN, deterministic first-point FPS, and interpolation during tracing, so the
generated model contains no custom operators. It accepts one preprocessed,
fixed-size point cloud as `points` with shape `[1, 1024, 3]` and returns:

| Output | Shape | Meaning |
| --- | --- | --- |
| `point_normals` | `[1, 1024, 3]` | Raw normal vector for every point |
| `cloud_normal` | `[1, 3]` | `normalize(mean(point_normals))` |

The point count is static in the exported graph and must be divisible by 8;
use `1024` for the bundled checkpoint. Every input must contain exactly that
many points: downsample larger patches and repeat points in smaller patches
before inference. Then preprocess the cloud exactly as the training dataset
does: select the local patch, subtract its mean, then divide by its maximum
point norm.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/export_onnx.py \
  msecnet_best/checkpoints/best.pt \
  --out msecnet_best/checkpoints/best.onnx \
  --num-points 1024
```

The exporter checks the resulting graph with the `onnx` Python package. Install
that package in the `point2normal` environment if it is not already available.

## Prepare a Pseudo-OBB Dataset

`prepare_pseudo_obb_dataset.py` is the exact data-preparation script from the
historical checkpoint commit. It creates thin pseudo-OBB prisms and a
car-model-disjoint `split_by_car_model.json` for the legacy training objective.

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_best/prepare_pseudo_obb_dataset.py \
  data/fuelcap_pass_20260721_9211 \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb \
  --max-points 4096 \
  --obb-expand 2.0 \
  --obb-half-depth-m 0.005
```

## Historical Checkpoint Reference

written to `msecnet_best/checkpoints/inference_test/` unless `--out` is given.
The bundled checkpoint was trained on the earlier 5873-sample dataset. Its
default inference command intentionally continues to evaluate that historical
dataset and is only a reference baseline:
written to `msecnet_best/checkpoints/inference_test/` unless `--out` is given.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py
```

Equivalent explicit historical command:

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  msecnet_best/checkpoints/best.pt \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_v4_fuelcap_pass_20260717_manual3d_pseudo_obb/split_by_car_model.json \
  --split-name test
```

## Train on the 9211 Dataset

This preserves the historical sign-invariant objective while training on the
newly prepared 9211-source pseudo-OBB dataset. A fresh run is not bit-for-bit
deterministic because the historical script uses time-based training sampling.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/train.py \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/split_by_car_model.json \
  --steps 70000 --bs 24 --max-points 1024 \
  --radius 0.3 --aug-deg 45 \
  --snapshot-every 1000 \
  --out msecnet_best/out/pseudo_obb_9211_legacy_v1
```

Important historical settings encoded in the checkpoint are `max_points=1024`,
`aug_deg=45`, variable-size point batches, and the loss
`1 - dot(normalize(point_prediction), target)^2`.

## Test the 9211 Checkpoint

After training, evaluate only the held-out 9211 test split. The report and CSV
are written under the new checkpoint directory.

```bash
conda run --no-capture-output -n point2normal python msecnet_best/infer.py \
  msecnet_best/out/pseudo_obb_9211_legacy_v1/best.pt \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/labels_manual3d.npz \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/clouds \
  --centers data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/anchors_manual3d.json \
  --split data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb/split_by_car_model.json \
  --split-name test
```

## Web Visualize 9211 Test Predictions

Launch the browser viewer for the held-out test split. The launcher runs the
same inference command above, writes `report.json` under the checkpoint's
`inference_test/` directory, and starts the read-only `web_label` viewer.

```bash
cd /data2/shendu/code/ruoyu/train_point2normal
bash web_label/launch_msecnet_test_viewer.sh
```

Open `http://127.0.0.1:8765` in a browser. In the viewer, the red arrow is the
manual target normal and the orange arrow is the MSECNet prediction. The
per-sample axial angle error is shown in the information panel. Because this
legacy model is sign-invariant, the viewer displays the equivalent prediction
direction nearest the target normal. This view is read-only and cannot change
manual labels.

Pass a checkpoint and port to inspect another run or avoid a port conflict:

```bash
bash web_label/launch_msecnet_test_viewer.sh PATH_TO_BEST_PT 8766
```

## Visualize a 9211 Pseudo-OBB Patch

This exports a PLY containing gray source context, blue training points, and a
yellow target normal. Omit `--file` to export the first validation sample.

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_best/export_pseudo_obb_ply.py \
  data/msecnet_best_fuelcap_pass_20260721_9211_manual3d_pseudo_obb \
  --source-root data/fuelcap_pass_20260721_9211 \
  --raw-points 50000
```

Each new training run writes these files to its `--out` directory:

| Path | Contents |
| --- | --- |
| `metrics.csv` | Fixed-column CSV for training loss, learning rate, and validation metrics |
| `dashboard.png` | Training loss, validation axial angle error, `<=10 deg` accuracy, and learning rate |
| `run.json` | Command-line arguments, sample counts, aggregation, and loss definition |
| `last.pt`, `best.pt` | Latest validation checkpoint and lowest-validation-error checkpoint |
| `snapshots/step_*.pt` | Periodic checkpoints controlled by `--snapshot-every` |
