import os
import random

import numpy as np
import torch
from PIL import Image

from deepcompressor.utils.common import hash_str_to_int

__all__ = ["get_control"]


def update_mask(mask: np.ndarray, x: int, y: int, radius: int | float):
    mask = mask.copy()
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if (j - x) ** 2 + (i - y) ** 2 <= radius**2:
                mask[i, j] = True
    return mask


def generate_mask(
    masked_ratio_range: tuple[int, int], size: int | tuple[int, int], seed: int | None = None, eps=1e-2
) -> np.ndarray:
    if seed is not None:
        random.seed(seed)
    masked_ratio = random.randint(masked_ratio_range[0], masked_ratio_range[1]) / 100
    if isinstance(size, int):
        size = (size, size)
    assert len(size) == 2
    height, width = size
    mask = np.zeros((height, width), dtype=bool)
    while True:
        radius = random.randint(16, min(height, width) // 2)
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        new_mask = update_mask(mask, x, y, radius)
        if new_mask.sum() / (height * width) <= masked_ratio + eps:
            mask = new_mask
            if mask.sum() / (height * width) >= masked_ratio - eps:
                break
    return mask


def center_crop_and_resize(image: Image.Image, target_size: int | tuple[int, int]) -> Image.Image:
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    else:
        assert len(target_size) == 2
    target_width, target_height = target_size

    width, height = image.size
    if width / height > target_width / target_height:
        new_width = height * target_width / target_height
        left = round((width - new_width) / 2)
        right = round(left + new_width)
        image = image.crop((left, 0, right, height))
    elif width / height < width / height:
        new_height = width * target_height / target_width
        top = round((height - new_height) / 2)
        bottom = round(top + new_height)
        image = image.crop((0, top, width, bottom))
    width, height = image.size
    if width != target_width or height != target_height:
        image = image.resize((target_width, target_height), Image.Resampling.BICUBIC)
    return image


def get_control(  # noqa: C901
    task: str,
    images: Image.Image | list[Image.Image],
    names: str | list[str] | None = None,
    data_root: str | None = None,
    device: str | torch.device = "cuda",
    **kwargs,
) -> Image.Image | list[Image.Image] | tuple[Image.Image, Image.Image] | tuple[list[Image.Image], list[Image.Image]]:
    size = kwargs.get("size", 1024)
    if isinstance(size, int):
        size = (size, size)
    assert len(size) == 2
    image_batch = [images] if isinstance(images, Image.Image) else images
    if isinstance(names, str):
        names = [names]

    if task == "canny-to-image":
        processor = kwargs.get("processor", None)

        control_images = []
        for i, image in enumerate(image_batch):
            if data_root is not None and names is not None:
                data_path = os.path.join(data_root, "canny_images", f"{names[i]}.png")
                if os.path.exists(data_path):
                    control_images.append(Image.open(data_path))
                    continue
            if processor is None:
                from controlnet_aux import CannyDetector

                processor = CannyDetector()
            image = center_crop_and_resize(image, size)
            control_image = processor(
                image, low_threshold=50, high_threshold=200, detect_resolution=max(size), image_resolution=max(size)
            )
            control_images.append(control_image)
        if isinstance(images, Image.Image):
            return control_images[0]
        return control_images
    elif task == "depth-to-image":
        processor = kwargs.get("processor", None)
        control_images = []
        for i, image in enumerate(image_batch):
            if data_root is not None and names is not None:
                data_path = os.path.join(data_root, "depth_images", f"{names[i]}.png")
                if os.path.exists(data_path):
                    control_images.append(Image.open(data_path))
                    continue
            if processor is None:
                from image_gen_aux import DepthPreprocessor

                processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)
            image = center_crop_and_resize(image, size)
            control_image = processor(image.convert("RGB"))[0].convert("RGB")
            control_images.append(control_image)
        if isinstance(images, Image.Image):
            return control_images[0]
        return control_images
    elif task == "inpainting":
        control_images, mask_images = [], []

        for i, image in enumerate(image_batch):
            name = None if names is None else names[i]
            if data_root is not None and name is not None:
                cropped_image_path = os.path.join(data_root, "cropped_images", f"{name}.png")
                mask_path = os.path.join(data_root, "mask_images", f"{name}.png")
                if os.path.exists(cropped_image_path) and os.path.exists(mask_path):
                    control_images.append(Image.open(cropped_image_path).convert("RGB"))
                    mask_images.append(Image.open(mask_path))
                    continue

            image = center_crop_and_resize(image, size)
            control_images.append(image.convert("RGB"))
            if names is not None:
                seed = hash_str_to_int(names[i])
            else:
                seed = None

            mask = generate_mask((5, 60), size, seed=seed)
            mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
            mask_images.append(mask_image)
        if isinstance(images, Image.Image):
            return control_images[0], mask_images[0]
        return control_images, mask_images
    else:
        raise ValueError(f"Unsupported task: {task}")
