import argparse
import os

import yaml
from tqdm import tqdm

from ...utils import get_control
from . import get_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmarks", type=str, nargs="*", default=["COCO", "DCI", "MJHQ"])
    parser.add_argument("--max-dataset-size", type=int, default=-1)
    parser.add_argument("--dump-root", type=str, default="benchmarks")
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--prompts-only", action="store_true")
    parser.add_argument("--controls", type=str, nargs="*", default=["canny-to-image", "depth-to-image", "inpainting"])
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--chunk-step", type=int, default=1)
    args = parser.parse_args()

    if "depth-to-image" in args.controls:
        from image_gen_aux import DepthPreprocessor

        processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf").to("cuda")

    for benchmark in args.benchmarks:
        dataset = get_dataset(
            benchmark,
            max_dataset_size=args.max_dataset_size,
            return_gt=True,
            chunk_start=args.chunk_start,
            chunk_step=args.chunk_step,
        )
        prompts = {}
        benchmark_root = os.path.join(args.dump_root, benchmark, f"{dataset.config_name}-{dataset._unchunk_size}")
        for row in tqdm(dataset, desc=f"Dumping {dataset.config_name}"):
            prompts[row["filename"]] = row["prompt"]
            if not args.prompts_only:
                image = row.get("image", None)
                if image is not None:
                    image_root = os.path.join(benchmark_root, "images")
                    os.makedirs(image_root, exist_ok=True)
                    if args.copy_images:
                        image.save(os.path.join(image_root, row["filename"] + ".png"))
                    else:
                        ext = os.path.basename(row["image_path"]).split(".")[-1]
                        os.symlink(
                            os.path.abspath(os.path.expanduser(row["image_path"])),
                            os.path.abspath(os.path.expanduser(os.path.join(image_root, row["filename"] + f".{ext}"))),
                        )
                    if "canny-to-image" in args.controls:
                        canny_root = os.path.join(benchmark_root, "canny_images")
                        os.makedirs(canny_root, exist_ok=True)
                        canny = get_control("canny-to-image", image)
                        canny.save(os.path.join(canny_root, row["filename"] + ".png"))
                    if "depth-to-image" in args.controls:
                        depth_root = os.path.join(benchmark_root, "depth_images")
                        os.makedirs(depth_root, exist_ok=True)
                        depth = get_control("depth-to-image", image, processor=processor)
                        depth.save(os.path.join(depth_root, row["filename"] + ".png"))
                    if "inpainting" in args.controls:
                        mask_root = os.path.join(benchmark_root, "mask_images")
                        cropped_image_root = os.path.join(benchmark_root, "cropped_images")
                        os.makedirs(mask_root, exist_ok=True)
                        os.makedirs(cropped_image_root, exist_ok=True)
                        cropped_image, mask_image = get_control("inpainting", image, names=row["filename"])
                        cropped_image.save(os.path.join(cropped_image_root, row["filename"] + ".png"))
                        mask_image.save(os.path.join(mask_root, row["filename"] + ".png"))

        if args.chunk_step == 1:
            os.makedirs(benchmark_root, exist_ok=True)
            with open(os.path.join(benchmark_root, "prompts.yaml"), "w") as f:
                yaml.dump(prompts, f)
