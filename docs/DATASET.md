# SUS dataset interface

## Required files

For each sample ID `<id>` listed in a split file, the loader reads:

- `seperated_images/<id>_rgb.jpg` as RGB input;
- `seperated_images/<id>_th.jpg` as thermal input and converts it to three channels;
- `labels/<id>.png` as the semantic target.

The directory name `seperated_images` follows the supplied dataset layout and is intentionally retained despite the spelling.

## Optional files

The following historical auxiliary targets are optional:

- `boundary/<id>.png`;
- `binary/<id>.png`;
- `SDF/<id>.png`;
- `weight/<id>.png`;
- `domain_mapping.json`.

When an auxiliary image is absent, the loader returns a zero-valued image with the same spatial size. When `domain_mapping.json` is absent, `domain_gt` defaults to zero.

## Dense text features

`dense_text_feature_path` should point to a PyTorch dictionary. A feature can be indexed by either `<id>_rgb.jpg` or `<id>`. Missing entries are replaced by a zero vector of `dense_text_feature_dim`, after which S2S uses its checkpointed fallback scene feature.

## Label IDs

| ID | Class |
|---:|---|
| 0 | unlabeled |
| 1 | road |
| 2 | sidewalk |
| 3 | person |
| 4 | car |
| 5 | bicycle |

By default, class 0 is excluded from mIoU and mF1 averages but remains part of the model output.
