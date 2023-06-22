import math
import os
import torch
from diffusers import UNet2DModel
from scheduling import ScheduleTypes, CustomScheduler, PredTypes
import numpy as np
from PIL import Image
from tqdm import tqdm
from train_utils import make_model, freeze_model
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def batch_sample(model, schedule, n_steps, n_images, batch_size, seed, save_path, timestep_type, pred_type):
    n_iterations = math.ceil(n_images / batch_size)
    gen = torch.Generator(device)
    gen.manual_seed(seed)
    noise_scheduler = CustomScheduler(device, schedule)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model.eval()
    j = 0
    for i in tqdm(range(n_iterations), position=0, desc="Batch", leave=False, colour='green', ncols=80):
        images = noise_scheduler.denoise(model, (3, 32, 32), gen, n_steps, batch_size, pred_type=pred_type, timestep_type=timestep_type)

        for im in images:
            im = (im * 255).astype(np.uint8)
            im = Image.fromarray(im)
            im.save(f'{save_path}{j}.png')
            j += 1


model = UNet2DModel.from_pretrained('c10/distill-v/3_30', device_map="auto")
freeze_model(model)
#model = make_model(32, "cuda")
#model.load_state_dict(torch.load("c10/v_base/30/b.pkl"))
#model = model.to(device)
batch_sample(model, ScheduleTypes.COSINE, 64, 16, 16, 321, 'show2-vdistill3/', "continuous", PredTypes.v)