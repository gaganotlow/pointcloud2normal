# MSECNet Ball RGB Fusion

This is an isolated experiment. It reads the existing `msecnet_ball` dataset
and its original source images without modifying either one.

The two inputs are intentionally separate:

- **geometry branch**: the existing center-anchored 8 cm point ball, processed
  by MSECNet with its radial-distance feature and radial weighted pooling;
- **RGB branch**: a crop from the original image centered on the projection of
  the reviewed 3D center. The crop size is set from the projected ball radius;
- **late fusion**: pooled geometry and image features are concatenated to
  regress the same camera-oriented 3D normal target.

No RGB value is assigned to a 3D point, so imperfect RGB/depth registration
cannot create an incorrect point-to-pixel correspondence. It still requires
that the source image and point cloud belong to the same sample.

The geometry head and RGB head both have auxiliary normal losses. During
training, `--rgb-dropout 0.20` removes the RGB feature for 20% of samples so
the fused model retains a meaningful point-cloud fallback. The default
`--aug-deg 0` is deliberate: arbitrary 3D coordinate rotation cannot be
applied consistently to a fixed camera image.

## Train

Run from the project root:

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/train.py \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260721_9211 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
  --out msecnet_ball_addRGB/out/rgb_fusion_v1
```

The output directory must be new or empty. The `best.pt` checkpoint contains
the exact crop and fusion schema; it is incompatible with point-only models.

## Evaluate

```bash
conda run --no-capture-output -n point2normal python \
  msecnet_ball_addRGB/infer.py \
  msecnet_ball_addRGB/out/rgb_fusion_v1/best.pt \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/labels_manual3d.npz \
  data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/clouds \
  --source-root data/fuelcap_pass_20260721_9211 \
  --centers data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/anchors_manual3d.json \
  --split data/msecnet_ball_v1_fuelcap_pass_20260721_9211_manual3d_r08/split_by_generalization_group.json \
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
