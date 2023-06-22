import torch
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel
from PIL import Image
import numpy as np
import os
from scheduling import CustomScheduler


def make_grid(images, rows, cols):
    w, h, r = images[0].shape
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, image in enumerate(images):
        im = (image * 255).astype(np.uint8)
        im = Image.fromarray(im)
        grid.paste(im, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, model, gen, timestep_number, schedule, pred_type, timestep_type, device):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    noise_scheduler = CustomScheduler(device, schedule)
    model.eval()
    images = noise_scheduler.denoise(model, (3, 32, 32), gen, timestep_number, config.eval_batch_size, pred_type,
                                     timestep_type=timestep_type)

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    return image_grid


def get_pretrained(path, device):
    model = UNet2DModel.from_pretrained(path)#torch.load(path,map_location=device)#UNet2DModel.from_pretrained(path)
    #model = model.to(device)
    return model


def make_model(image_size, device):
    model = UNet2DModel(
        sample_size=image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",  # a regular ResNet upsampling block
            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        ),
    )
    model = model.to(device=device)
    return model


preprocess = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["img"]]
    return {"img": images}


def get_dataloader(dataset_name, batch_size):
    dataset = load_dataset(dataset_name, split="train")
    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader


def compute_sigma(alpha):
    return torch.sqrt(1 - alpha ** 2)


def compute_alpha(t):
    return torch.cos((0.5 * torch.pi * t + 0.008) / 1.008)


def batch_first(a):
    return a.permute(3, 0, 1, 2)


def batch_last(a):
    return a.permute(1, 2, 3, 0)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False