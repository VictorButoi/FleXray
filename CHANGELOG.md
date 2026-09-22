# Release notes

## 1.0.2

FleXray development, issues, pull requests, and package releases now live in the
fresh public repository at https://github.com/VictorButoi/FleXray. Its history
starts with a single `Initial commit`; private development history is not part
of this repository. The package remains `flexray`, imported as `fxr`.

This release includes the cleanup already completed before the public import:

- The legacy `fxr.inference.convert_omniax_run` conversion path is retired.
  Native checkpoint-to-bundle and browser ONNX exporters are maintained in
  OmniAX; model/config validation and checkpoint readers remain in FleXray.
- `GeneratedTrainingDataset` and the separate `generated` modality are removed.
  FluXray and other generated 2D image/channel-mask samples use the X-ray data
  path. CT samples continue to render through the DRR path.
- The `UNet_FleXray_default` training config alias is removed. Use `base` or
  an explicit training configuration.
- Website source, assets, builders, browser tests, and deployment belong to
  FleXray-website. The canonical site is https://flexray.csail.mit.edu/.
- Training uses `nanodrr>=0.1.5`, as required by the audited training code.

The supported commands remain `flexify`, `fxr-dataset`, `fxr-train`,
`fxr-submit`, `fxr-render`, `fxr-protocol`, and `fxr-mcp`. The default install
remains inference-only; training dependencies are installed with
`pip install 'flexray[train]'`. This public import introduces no additional
package API, label-space, storage-format, config-schema, or dependency-policy
changes beyond the cleanup above.

### Preserved endpoints and artifacts

- Current GitHub source and documentation paths use the public repository.
- Existing PyPI 1.0.0 and 1.0.1 files, hashes, and provenance records remain
  unchanged. They will not be rebuilt or uploaded again.
- The custom-domain website, demo, attribution pages, assets, and current
  Colab sample URLs remain available through FleXray-website.
- Hugging Face model/data repositories and published revisions are unchanged.

### Intentional retirements

- Historical commits, pull requests, Actions runs, tags, and GitHub release
  downloads are no longer available at the original FleXray repository address.
- `https://victorbutoi.github.io/FleXray/` and its assets are retired, including
  homepage links embedded in old PyPI releases. Use the custom domain above.
- Links in old PyPI provenance records to historical GitHub commits or runs may
  stop resolving; the provenance records themselves remain unchanged.
