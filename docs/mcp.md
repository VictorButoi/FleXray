# `fxr.mcp`

The MCP server exposes FleXray's public pretrained-inference and
protocol-introspection surfaces to Model Context Protocol clients such as
Claude Code and Claude Desktop over local stdio. Tool behavior wraps the same
contracts as `flexify` and `fxr.protocols`; the server deliberately does not
expose training launch, DRR rendering, config validation, or image-overlay
rendering.

`fxr-mcp` requires the MCP extra:

```bash
python -m pip install "flexray[mcp]"
# Repository checkout:
uv sync --extra mcp --extra test
```

The base inference install reports the missing extra without a traceback.

## Tools

All tools return JSON. Tool errors carry the underlying exception message
(`FileNotFoundError`, `NotADirectoryError`, `TypeError`, `ValueError`) without
a traceback.

### `segment_image`

Segments one image file or every supported image in a directory
(`.bmp/.jpeg/.jpg/.png/.tif/.tiff`, plus `.dcm`/`.dicom` and extension-less
files with a `DICM` header, non-recursive; DICOM needs the `dicom` or `full`
extra). Writes the same NumPy
artifacts as `flexify` — `{stem}_logits.npy`, `{stem}_probabilities.npy`
(float32, `CxHxW`), `{stem}_masks.npy` (uint8, `CxHxW`) — with the same
collision-deduped stems, and returns per-label statistics computed from the
prediction. Statistics are taken inside `content_box`, the region of the
bundle's model canvas (256x256 by default) the oriented input occupies after
pad-to-square and resize. `content_box` is `[top, left, bottom, right]`, with
exclusive bottom/right bounds rounded to canvas pixels. Subpixel footprints
are expanded to at least one pixel; bilinear interpolation can mix input and
padding along the boundary. `pixel_count` counts canvas pixels and
`pixel_fraction` divides by the area of this rounded content region. Saved
artifacts still contain the full canvas. `content_box` is `null` for DICOM
inputs, whose statistics cover the whole canvas.

| Argument | Default | Meaning |
| --- | --- | --- |
| `input_path` | required | Image file or directory of image files. |
| `output_dir` | required | Artifact directory, created if missing. |
| `model_id` | `VictorButoi/flexray` | Hugging Face repo id with FleXray bundle artifacts, or a list of repo ids averaged as an ensemble. A repo with an `ensemble.json` loads its flagship bundle by default. |
| `revision` | `None` | Hugging Face branch, tag, or commit id. |
| `subfolder` | `None` | Bundle directory inside a single `model_id`, e.g. `members/flux000`. |
| `ensemble` | `false` | Average every member declared by a single `model_id`'s `ensemble.json` (the published five-model ensemble). |
| `label` | `None` | Restrict artifacts and statistics to one output label. |
| `threshold` | `0.5` | Probability threshold for uint8 masks, in `[0, 1]`. |
| `tta_samples` | `1` | Total TTA forward passes per image; `<= 1` disables TTA. |

Example return:

```json
{
  "model_id": "VictorButoi/flexray",
  "revision": null,
  "subfolder": null,
  "ensemble": false,
  "threshold": 0.5,
  "tta_samples": 1,
  "image_count": 1,
  "results": [
    {
      "image_path": "/data/slide_00.png",
      "artifacts": {
        "logits": "/out/slide_00_logits.npy",
        "probabilities": "/out/slide_00_probabilities.npy",
        "masks": "/out/slide_00_masks.npy"
      },
      "content_box": [0, 0, 256, 256],
      "labels": [
        {
          "label": "femurs",
          "channel": 11,
          "pixel_count": 1284,
          "pixel_fraction": 0.0196,
          "mean_probability": 0.041,
          "max_probability": 0.973
        }
      ]
    }
  ]
}
```

### `list_models` and `describe_model`

`list_models` returns the static registry of published FleXray bundles without
touching the network: every entry names the repo (`model_id`) and the bundle
directory inside it (`subfolder`); any Hugging Face repo id containing FleXray
bundle artifacts is also accepted by `segment_image` and `describe_model`.
`describe_model(model_id, revision, subfolder, ensemble)` loads a bundle
(first call downloads it from Hugging Face) and reports its exact output
labels with channel ids, `probability_mode`, and preprocessing metadata.

### Protocol tools

- `list_protocols` — packaged protocol names.
- `describe_protocol(name)` — channel-ordered labels; `background` is
  channel 0.
- `list_datasets` — packaged dataset spec names with declared aliases.
- `describe_dataset(name)` — native stored mask ids, per-protocol label
  aliases, and per-protocol drop labels; accepts dataset aliases.
- `explain_mapping(protocol_name, dataset_name)` — the compiled training
  lookup table: dense `label_lut` (`-1` marks native-id gaps) plus one row per
  declared native id with its `rule` (`background`, `identity`, `alias`,
  `drop`), destination protocol channel, and label.

Registry tools accept config names only, never paths. The implementations are
shared with the `fxr-protocol` command and live in `fxr.protocols.introspect`.

The MCP tool returns channel names in each result's `labels` field. The CLI's
`label_names.json` sidecar and skip-and-continue batch handling apply to
`flexify`; `segment_image` stops on an invalid input and does not currently expose
the Python API's `seed` argument.

## Running the Server

`fxr-mcp` runs on stdio until the client disconnects. Models stay cached in the
server process keyed by `(model_id, revision, subfolder, ensemble)`.

`--device` defaults to `auto`: CUDA when available, otherwise CPU.

```bash
# Claude Code
claude mcp add flexray -- fxr-mcp
# From a repository checkout without an activated environment:
claude mcp add flexray -- /path/to/FleXray/.venv/bin/fxr-mcp

# MCP Inspector
npx @modelcontextprotocol/inspector fxr-mcp
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "flexray": {
      "command": "fxr-mcp"
    }
  }
}
```

## Programmatic API

`fxr.mcp.server.create_server()` returns the configured `FastMCP` instance
(requires the `mcp` extra). The tool implementations are plain functions —
`fxr.mcp.segment_image`, `fxr.mcp.describe_protocol`, and the rest of
`fxr.mcp.__all__` — importable without the MCP SDK and without loading torch
until a model is needed:

```python
from fxr.mcp import describe_protocol, segment_image

describe_protocol("all_structures_flexray_v4")["labels"][:3]
segment_image("xray.png", "out/", label="femurs")
```

## Runtime Boundary

The server exposes only the public pretrained-inference path (Hugging Face
bundles) and packaged protocol metadata. Training launch, submitit
submission, online CT-to-DRR rendering, and legacy run-dir loading are outside
the MCP surface.
