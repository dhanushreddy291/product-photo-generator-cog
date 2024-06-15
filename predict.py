# Prediction interface for Cog
import os
import shutil
from typing import List

import torch
from cog import BasePredictor, Input, Path
from diffusers import DiffusionPipeline
from diffusers.utils import load_image
from PIL import ImageOps
from transparent_background import Remover

MODEL_NAME = "yahoo-inc/photo-background-generation"
MODEL_CACHE = "model-cache"
device = "cuda"
float_datatype = torch.float16


class Predictor(BasePredictor):
    def setup(self) -> None:
        self.pipe = DiffusionPipeline.from_pretrained(
            MODEL_NAME,
            cache_dir=MODEL_CACHE,
            torch_dtype=float_datatype,
            custom_pipeline=MODEL_NAME,
            local_files_only=True,
        ).to(device)

    def resize_with_padding(self, img, expected_size):
        img.thumbnail((expected_size[0], expected_size[1]))
        delta_width = expected_size[0] - img.size[0]
        delta_height = expected_size[1] - img.size[1]
        pad_width = delta_width // 2
        pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return ImageOps.expand(img, padding)

    def loadImage(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="A Shoe on a marble podium, product photography, high resolution",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="3d, cgi, render, bad quality, normal quality",
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of inference steps",
            ge=20,
            le=50,
            default=30,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        image: Path = Input(
            description="Input image for Generating the Background",
            default=None,
        ),
        controlnet_conditioning_scale: float = Input(
            description="Controlnet Conditioning Scale",
            default=1.0,
            le=3.0,
            ge=1.0,
        ),
    ) -> List[Path]:
        if seed is None:
            seed = int.from_bytes(os.urandom(3), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator(device).manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "guess_mode": False,
        }

        img = self.loadImage(image)
        img = self.resize_with_padding(img, (512, 512))

        remover = Remover()
        remover = Remover(mode="base")

        # Get foreground mask
        fg_mask = remover.process(img, type="map")
        mask = ImageOps.invert(fg_mask)

        with torch.autocast(device):
            output = self.pipe(
                **common_args, mask_image=mask, image=img, control_image=mask
            )

        output_paths = []
        for i, im in enumerate(output.images):
            output_path = f"/tmp/out-{i}.png"
            im.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths

