# FleXray Model Zoo

Public FleXray models live in one Hugging Face model repository,
[`VictorButoi/flexray`](https://huggingface.co/VictorButoi/flexray),
and load with `FleXraySegmenter.from_pretrained` / `flexify`. The repository
root holds the model card and `ensemble.json`; every bundle sits under
`members/`:

| Subfolder | FluXray proportion | Role |
| --- | --- | --- |
| `members/flux0375` | 0.375 | Flagship; loaded by default (`FLAGSHIP_SUBFOLDER`). |
| `members/flux000` | 0.0 | Ensemble member. |
| `members/flux025` | 0.25 | Ensemble member. |
| `members/flux050` | 0.5 | Ensemble member. |
| `members/flux075` | 0.75 | Ensemble member. |

All five share one architecture, label schema, preprocessing contract, and
training recipe (`fxr/configs/training/base.yml`), differ only in the FluXray
proportion of the training mix, and ship the EMA weights of `last.pt`. Weights
are `CC-BY-NC-4.0`; code is MIT.

`flexify --ensemble` and `FleXraySegmenter.from_pretrained(ensemble=True)`
average all five (the website demo's High / X-High quality modes);
`--subfolder members/flux000` / `from_pretrained(subfolder="members/flux000")`
load one member. The browser demo pins one repository revision for all five
ONNX models in the [public demo manifest](https://flexray.csail.mit.edu/demo/demo_manifest.json).

Repository files:

- `README.md` model card
- `ensemble.json` — `flagship` subfolder plus the `members` list
- `members/<name>/model.safetensors`
- `members/<name>/config.yml`
- `members/<name>/label_schema.json`
- `members/<name>/preprocessing.json`
- `members/<name>/checksums.json`
- `members/<name>/onnx/flexray-<name>-256-fp16.onnx` — published ONNX model
  consumed by the website demo
