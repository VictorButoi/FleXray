# Harmonize and train on your own dataset

This walkthrough shows FleXray's label harmonizer and config system end to end
on a tiny synthetic hip X-ray dataset. Every command runs on CPU in seconds.

Three files describe the dataset:

| File | Role |
| --- | --- |
| `make_synthetic_dataset.py` | Writes six 64x64 PNG image/mask pairs and a packaging manifest (`dataset.yml`). Replace it with your own images and masks. |
| `CustomHips.yml` | **Dataset spec**: the dataset's native mask ids and how they map into the FleXray protocol (`hip_left`/`hip_right` merge into `hips`, `implant` drops to background, `femurs` maps by name). |
| `train_custom.yml` | Training config: protocol, model, data source, loss route, the `base_light` augmentation preset (the default `Xray_base` chain needs inputs of at least 224 px per axis), and `?` placeholders filled at launch. |

## Run it

```bash
cd examples/custom_dataset
pip install "flexray[train]"              # once

python make_synthetic_dataset.py --output work/data
fxr-protocol compile --dataset CustomHips.yml
fxr-dataset validate work/data/dataset.yml
fxr-dataset pack work/data/dataset.yml "$PWD/work/packed/CustomHips"
fxr-train train_custom.yml \
  --set data.Xray.CustomHips.path="$PWD/work/packed/CustomHips" \
  --set data.Xray.CustomHips.dataset_spec="$PWD/CustomHips.yml" \
  --set log.root="$PWD/work/runs" --device cpu --dry-run --smoke-data
```

`fxr-protocol compile` prints the compiled lookup table so you can check the
harmonization before training:

```text
CustomHips -> all_structures_flexray_v4
native_id  native_label  rule        channel  protocol_label
        0  background    background        0  background
        1  hip_left      alias            54  hips
        2  hip_right     alias            54  hips
        3  femurs        identity         11  femurs
        4  implant       drop              0  background
label_lut: [0, 54, 54, 11, 0]
```

The dry run validates the config, the package, and the label mapping without
training. Drop `--dry-run --smoke-data` to train two tiny epochs (WandB runs in
offline mode). The run directory then holds `config.yml`,
`augmentations/xray.yml` (the resolved augmentation chain), and
`checkpoints/last.pt`.

## How the pieces fit

- The **protocol** (`all_structures_flexray_v4`, 63 labels) is the ground-truth
  label space; `fxr-protocol show all_structures_flexray_v4` lists it. The
  released model's 61 output channels follow its bundle `label_schema.json`
  (the two aggregate spine labels are omitted), so protocol channel numbers
  are not output channel indices.
- A **dataset spec** maps a dataset's native ids into the protocol with
  `protocol_label_aliases` (many-to-one merges) and `protocol_drop_labels`
  (ignored structures). Names not listed must be protocol names (identity).
  The packaged specs for FleXray's own training data live in
  `fxr/configs/datasets/`; yours can stay next to your data.
- `data.Xray.<name>.dataset_spec` points the training config at that spec. The
  spec's `dataset_name` and `stored_labels` must match the package exactly;
  readiness checks fail otherwise. Without a spec, a package's `stored_labels`
  must already use protocol names.
- `fxr-dataset` packs images and masks into the ThunderDB layout training reads.
  For real data, `fxr-dataset scaffold` writes the manifest from image and
  mask directories, and the optional `preprocessing:` block reproduces the
  released model's padding/resizing (see `docs/datasets.md`).

## Next steps

- Fine-tune the released model instead of training from scratch:
  `fxr-train train_custom.yml --init-from VictorButoi/flexray ...` keeps
  the full protocol; add `--replace-head` to fine-tune onto a different
  protocol (`docs/training.md`).
- Render DRR training data from your own CT volumes with `fxr-render`
  (`docs/camera.md`), or train the full recipe with `fxr-train --base base`.
