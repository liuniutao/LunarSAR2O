import argparse
import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.LunarS2OUNet import LunarS2OUNet


SUPPORTED_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
DEFAULT_CKPT = Path("ckpt/CKPTv1.pth")
PATCH_SIZE = 512
OVERLAP = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tiled SAR-to-optical inference on a single image or a folder."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input grayscale image path, or a folder containing grayscale images.",
    )
    parser.add_argument(
        "--output",
        default="outputs",
        help="Output directory for generated RGB images. Default: outputs",
    )
    parser.add_argument(
        "--ckpt",
        default=str(DEFAULT_CKPT),
        help=f"Checkpoint path. Default: {DEFAULT_CKPT.as_posix()}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"Tiled inference patch size. Default: {PATCH_SIZE}",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=OVERLAP,
        help=f"Overlap between neighboring patches. Default: {OVERLAP}",
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]

    image_files: List[Path] = []
    for extension in SUPPORTED_EXTENSIONS:
        image_files.extend(Path(path) for path in glob.glob(str(input_path / extension)))
        image_files.extend(
            Path(path) for path in glob.glob(str(input_path / extension.upper()))
        )
    return sorted(set(image_files))


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = LunarS2OUNet().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()
    return model


def build_weight_patch(patch_size: int, device: torch.device) -> torch.Tensor:
    win = np.linspace(0, 1, patch_size, dtype=np.float32)
    wx, wy = np.meshgrid(win, win)
    weight_patch = np.minimum(wx, 1 - wx) * np.minimum(wy, 1 - wy)
    return torch.from_numpy(weight_patch).unsqueeze(0).to(device)


def sliding_window_cnn(
    sar_img: Image.Image,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    overlap: int,
) -> Image.Image:
    sar_np = np.array(sar_img, dtype=np.float32) / 255.0
    height, width = sar_np.shape
    sar_tensor_full = torch.from_numpy(sar_np).unsqueeze(0).to(device)

    output_accum = torch.zeros(3, height, width, device=device)
    weight_accum = torch.zeros(1, height, width, device=device)
    weight_patch = build_weight_patch(patch_size, device)

    stride = patch_size - overlap
    if stride <= 0:
        raise ValueError("patch_size must be larger than overlap")

    with torch.no_grad():
        for y in tqdm(range(0, height, stride), desc="Tiled Inference", leave=False):
            for x in range(0, width, stride):
                y1 = min(y + patch_size, height)
                x1 = min(x + patch_size, width)
                y0 = max(0, y1 - patch_size)
                x0 = max(0, x1 - patch_size)

                patch = sar_tensor_full[:, y0:y1, x0:x1]
                patch_h, patch_w = patch.shape[1], patch.shape[2]
                if patch_h < patch_size or patch_w < patch_size:
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, patch_size - patch_w, 0, patch_size - patch_h),
                        mode="reflect",
                    )

                output = model(patch.unsqueeze(0))
                output = torch.clamp(output, 0, 1).squeeze(0)[:, :patch_h, :patch_w]
                weight_crop = weight_patch[:, :patch_h, :patch_w]

                output_accum[:, y0:y1, x0:x1] += output * weight_crop
                weight_accum[:, y0:y1, x0:x1] += weight_crop

    output_final = (output_accum / weight_accum.clamp(min=1e-6)).cpu().numpy()
    rgb_out = (output_final.transpose(1, 2, 0) * 255).round().astype(np.uint8)
    return Image.fromarray(rgb_out)


def process_images(
    image_paths: List[Path],
    output_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    patch_size: int,
    overlap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        sar_img = Image.open(image_path).convert("L")
        result = sliding_window_cnn(sar_img, model, device, patch_size, overlap)
        output_path = output_dir / f"{image_path.stem}_rgb{image_path.suffix}"
        result.save(output_path)
        print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    checkpoint_path = Path(args.ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    image_paths = collect_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No supported image files found in: {input_path}")

    model = load_model(checkpoint_path, device)
    process_images(
        image_paths=image_paths,
        output_dir=output_dir,
        model=model,
        device=device,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
