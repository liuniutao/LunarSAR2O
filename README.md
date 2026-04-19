# 3LunarS2O

Inference-only PyTorch release for Lunar SAR-to-optical image translation.

This repository is trimmed to a single public demo pipeline:

`raw SAR .mat -> Preprocess.m -> SAR .png -> infer_BigRS.py -> optical output`

## Repository contents

- `infer_BigRS.py`: tiled inference script
- `models/LunarS2OUNet.py`: inference network used in this release
- `models/blocks/conv_blocks.py`: shared convolution block used by the network
- `ckpt/CKPTv1.pth`: default checkpoint
- `data/Example input images/Preprocess.m`: converts one example raw SAR group from `.mat` to `.png`
- `data/Example input images/v1/`: public example raw SAR group and preprocessed PNGs
- `data/Example output images/`: generated optical outputs for the public example

## Environment

```bash
pip install -r requirements.txt
```
Tested inference environment: Python 3.8.19 in the local `pycnn` Conda environment.

## Demo pipeline

1. Preprocess the raw SAR `.mat` files in MATLAB:

```matlab
cd('data/Example input images')
Preprocess
```

By default, `Preprocess.m` converts the `v2/` group. If needed, edit `groupName` in the script.

2. Run inference:

```bash
python infer_BigRS.py --input "data/Example input images/v2" --output "data/Example output images/v2"
```

## Notes

- Input images are expected to be SAR images，with size at or above 600×600 pixel.
- Output images are saved as `{input_name}_rgb.{ext}`.
- This repository does not include training code.


## Minimal files still worth adding

- `LICENSE`
- A short note describing the `v1` example group and the source of the raw SAR data
- Optional comparison figures for the generated optical results
