import math
import os
import torch
from diffusers import UNet2DModel
from scheduling import ScheduleTypes, CustomScheduler
import numpy as np
from PIL import Image
from diffusers import DDPMPipeline
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def batch_sample(model, schedule, n_steps, n_images, batch_size, seed, save_path):
    n_iterations = math.ceil(n_images / batch_size)
    gen = torch.Generator(device)
    gen.manual_seed(seed)
    noise_scheduler = CustomScheduler(device, schedule)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    j = 0
    for i in tqdm(range(n_iterations)):
        images = noise_scheduler.denoise(model, (3, 32, 32), gen, n_steps, batch_size)

        for im in images:
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(f'{save_path}{j}.png')
            j += 1


#model = UNet2DModel.from_pretrained('./c10', device_map="auto")
model = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").unet
model = model.to(device)
batch_sample(model, ScheduleTypes.LINEAR, 1024, 5000, 16, 101, './teacher/')