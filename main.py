import spandrel_extra_arches
from spandrel import ImageModelDescriptor, ModelLoader
import torch
from pathlib import Path
import argparse
from PIL import Image
import numpy as np


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img) / 255

    return torch.from_numpy(img).unsqueeze(0).float().cuda()


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    print(f"The number of dimensions is: {tensor.ndim}")

    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"{tensor.shape} does not describe a BCHW tensor.")

        tensor = tensor.squeeze(0)

    assert tensor.ndim == 3, f"{tensor.shape} does not describe a CHW tensor"

    arr = tensor.float().cpu().clamp_(0, 1).numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = arr.round().astype(np.uint8)
    arr = arr[:, :, ::-1]

    return Image.fromarray(arr, "RGB")


def process(image: torch.Tensor, span_model: any) -> torch.Tensor:
    with torch.no_grad():
        upscaled_img = span_model(image)

        return upscaled_img


if __name__ == "__main__":
    print("Starting upscale.")

    spandrel_extra_arches.install()

    model_path = Path.cwd().joinpath("upscale_models", "srformer_4x_stock.pth")
    path_to_img = Path.cwd().joinpath("input_images", "original.png")

    model = ModelLoader().load_from_file(model_path)

    assert isinstance(model, ImageModelDescriptor)

    model.scale = 4
    model.cuda().eval()

    image = pil_to_tensor(img=Image.open(path_to_img))
    image = process(image=image, span_model=model)
    image = tensor_to_pil(tensor=image)

    image.save(Path.cwd().joinpath("output_images", "upscaled.png"))
