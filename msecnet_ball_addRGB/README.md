# MSECNet Ball RGB Fusion

This is an isolated experiment. It reads the existing `msecnet_ball` dataset
and its original source images without modifying either one.

The two inputs are intentionally separate:

- **geometry branch**: the existing center-anchored 8 cm point ball, processed
  by MSECNet with its radial-distance feature and radial weighted pooling;
- **RGB branch**: a camera-oriented crop localized by the detector's rotated
  OBB in the original image. The crop is not rotated, because the target normal
  is expressed in the camera coordinate frame. A pretrained DINOv2 ViT-S/14
  encoder produces its visual representation;
- **late fusion**: pooled geometry and image features are concatenated to
  regress the same camera-oriented 3D normal target.

No RGB value is assigned to a 3D point, so imperfect RGB/depth registration
cannot create an incorrect point-to-pixel correspondence. It still requires
that the source image and point cloud belong to the same sample.

The recommended protocol starts from a validated point-only `best.pt`, keeps
that complete MSECNet classifier frozen, and lets RGB make only a small gated
residual correction. This preserves the point-cloud result when RGB is not
useful. During training, `--rgb-dropout 0.20` removes the RGB feature for 20%
of samples. The default `--aug-deg 0` is deliberate: arbitrary 3D coordinate
rotation cannot be applied consistently to a fixed camera image.

## Cache OBBs

Run detection once for the labeled samples. The cache is resumable and is used
unchanged by both training and evaluation. A sample with no target OBB remains
in the JSON `skipped` list; training excludes it rather than falling back to a
reviewed-center crop.

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/detect_obb.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  --source-root data/fuelcap_pass_20260803_10847 \
  --obb-model ../fuelcap_6dpose/models/inner_obb_clean_v11m_0129/best.pt \
  --out data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --class-id 0 --conf 0.25 --batch-size 32
```

The RGB crop expands the detected rectangle by `--obb-crop-scale 1.4`, then
resizes the camera-oriented square to the model's input size. This preserves
only a small context margin around the detector-localized target.

## Visual Backbone

`dino_vits14` is the default RGB backbone. Its pretrained DINOv2 ViT-S/14
weights are loaded once from the local Torch Hub cache; the final two
Transformer blocks and the RGB projection head are fine-tuned. Use
`--dino-unfreeze-blocks 0` to freeze DINOv2 completely. DINOv2 requires an
image size divisible by 14, so the default OBB crop size is `224`. Unfrozen
DINOv2 blocks use `--dino-lr 1e-5`, while the newly initialized heads use the
main `--lr`; do not apply the head learning rate to DINOv2 itself.

The default `gated_residual` fusion predicts an RGB correction to the
geometry-only normal. For the recommended frozen-point protocol, the residual
norm is capped by `--max-rgb-correction`. `--initial-gate` starts the RGB path
at a small but trainable contribution; use only a light `--gate-penalty` so it
does not suppress every possible visual correction. `metrics.csv` and
`dashboard.png` include the validation mean gate.

## Train

Run from the project root:

The recommended V5 protocol does not use one global RGB vector for the whole
ball. It projects every sampled camera-space ball point into the same
camera-oriented OBB crop, samples DINO patch tokens at that location, and
fuses the resulting visual feature with that point's MSECNet feature. This
requires the OBB crop protocol and keeps the point-only checkpoint frozen.

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260803_10847 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --obb-detections data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --obb-crop-scale 3.5 \
  --image-backbone dino_vits14 --image-size 336 --dino-unfreeze-blocks 1 --dino-lr 3e-6 \
  --geometry-mode pretrained_point \
  --geometry-checkpoint msecnet_ball/out/center_ball_r08_oriented_20260803_v1/best.pt \
  --freeze-geometry --fusion-mode point_aligned_residual \
  --max-rgb-correction 0.05 --initial-gate 0.10 --gate-penalty 0.001 \
  --point-loss-weight 0 --geometry-loss-weight 0 --image-loss-weight 0.05 \
  --rgb-dropout 0.00 --steps 30000 --batch-size 16 --max-points 1024 \
  --out msecnet_ball_addRGB/out/rgb_fusion_dino_obb_20260803_v1
```

The output directory must be new or empty. The `best.pt` checkpoint contains
the exact crop, point-baseline initialization, residual limit, and fusion
schema; it is incompatible with point-only models. V5 fine-tunes only the
last DINO block with a low learning rate while keeping the point model frozen.
It writes no step-by-step weight snapshots.
For this dataset, `3.5` keeps 98.8% of projected ball points inside the OBB
crop on test; the historical global-fusion value `1.4` leaves nearly half of
the ball points outside and is deliberately rejected by V5.

Each validation writes the current metrics to `metrics.csv` and updates the
four-panel training plot in `dashboard.png`. The run also writes `run.json`,
`last.pt`, and `best.pt`; historical step-by-step weight snapshots are not
created.

## Evaluate

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/infer.py \
  msecnet_ball_addRGB/out/rgb_fusion_dino_obb_20260803_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260803_10847 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/split_by_generalization_group.json \
  --obb-detections data/msecnet_ball_v1_fuelcap_pass_20260803_10847_manual3d_r08/obb_inner_0129.json \
  --split-name test
```

Compare the held-out `mean_angular_error_deg` against the existing point-only
model, and inspect the geometry-only and image-only predictions in
`report.json` when fusion changes the result.

## CPU checks

```bash
conda run --no-capture-output -n point2normal python -m unittest \
  msecnet_ball_addRGB.test_data
```
