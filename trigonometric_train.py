import torch
from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
from diffusers import UNet2DModel

from PIL import Image
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from accelerate import Accelerator
import numpy as np

import copy
from tqdm.auto import tqdm
import os
from old_scheduling import TrigScheduler, ScheduleTypes


def train_loop_eps(config, student, optimizer, train_dataloader, lr_scheduler, teacher, schedule, timestep_type):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        logging_dir=os.path.join(config.output_dir, "logs")
    )

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    student, optimizer, train_dataloader, lr_scheduler, teacher, schedule = accelerator.prepare(
        student, optimizer, train_dataloader, lr_scheduler, teacher, schedule
    )

    global_step = 0
    K = 3
    starting_K = 0
    N = 512
    starting_epoch = 0
    for j in range(starting_K, K):
        noise_scheduler = TrigScheduler(device, schedule)

        # ts = torch.from_numpy(np.arange(0, N)[::-1].copy()).to(device)
        one = torch.tensor([1], device=device, dtype=torch.int64)
        # Add 2 elements at the beggining to evaluate t-1 and t-2
        # ts = torch.cat((one, one, ts), 0)
        # Timesteps for mid-steps e.g. t - 0.5/N
        ts2 = torch.from_numpy(np.arange(0, 2 * N)[::-1].copy()).to(device)
        ts2 = torch.cat((one, one, ts2), 0)
        if timestep_type == "discrete":
            ts2N = ts2
        else:
            ts2N = ts2/N
        # tsN = ts/N
        betas = noise_scheduler.get_betas(2 * N, ts2)
        alphas = 1 - betas

        # Student parameters = Teacher parameters
        if j > 1:
            student.load_state_dict(teacher.state_dict())
        # Now you train the model
        for epoch in range(starting_epoch, config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader))
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                #Sample data
                x = batch['img']  # Batch size last for timestep multiplication

                # Sample a random timestep for each image
                i = torch.randint(0, int(N), (x.shape[0],), device=x.device).long()
                i = torch.add(i, other=2)

                alpha_t = alphas[ts2[i]]
                sigma_t = compute_sigma(alpha_t).to(x.device)

                # Sample noise to add to the images
                epsilon = torch.randn(x.shape).to(x.device)

                xT = batch_last(x)
                epsilonT = batch_last(epsilon)

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                zT_t = (alpha_t * xT + sigma_t * epsilonT)
                z_t = batch_first(zT_t)

                # 2 Steps of DDIM of teacher
                alpha_t1 = alphas[ts2[i-1]]  # t - 0.5/N
                alpha_t2 = alphas[ts2[i-2]]  # t - 1/N
                sigma_t1 = compute_sigma(alpha_t1).to(x.device)
                sigma_t2 = compute_sigma(alpha_t2).to(x.device)

                eps_t = teacher(z_t, ts2N[i], return_dict=False)[0]
                epsT_t = batch_last(eps_t)

                xT_t = (zT_t - sigma_t * epsT_t) / alpha_t

                zT_t1 = alpha_t1 * xT_t + sigma_t1 * epsT_t
                z_t1 = batch_first(zT_t1)

                eps_t1 = teacher(z_t1, ts2N[i-1], return_dict=False)[0]
                epsT_t1 = batch_last(eps_t1)

                xT_t1 = (zT_t1 - sigma_t1 * epsT_t1) / alpha_t1

                zT_t2 = alpha_t2 * xT_t1 + sigma_t2 * epsT_t1


                x_tilde = (zT_t2 - (sigma_t2 / sigma_t) * zT_t) / (alpha_t2 - (sigma_t2 / sigma_t) * alpha_t)
                eps_tilde = (zT_t2 - alpha_t2 * x_tilde) / sigma_t2
                omega_t = torch.max(torch.tensor([torch.mean(alpha_t ** 2 / sigma_t ** 2), 1]))
                #i2 = (i//2).to(torch.long)
                #alpha_s = alphas[ts2[i2]]
                #sigma_s = compute_sigma(alpha_s)
                #zT_s = (alpha_t * xT + sigma_t * epsilonT)
                #z_s = batch_first(zT_s)

                with accelerator.accumulate(student):
                    # Predict the noise residual
                    #student(z_t, tsN[i], return_dict=False)[0]

                    eps_student = student(z_t, ts2N[i], return_dict=False)[0]
                    epsT_student = batch_last(eps_student)
                    #x_student = (zT_t - sigma_t * epsT_student) / alpha_t
                    '''x1 = x_student.detach().cpu().numpy().transpose()[0]
                    x2 = x_tilde.detach().cpu().numpy().transpose()[0]

                    plt.imshow(x1)
                    plt.show()
                    plt.imshow(x2)
                    plt.show()'''

                    #loss_x = F.mse_loss(x_tilde, x_student)
                    loss = omega_t * F.mse_loss(eps_tilde, epsT_student)
                    #loss = torch.maximum(loss_x, loss_eps)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(student.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                    evaluate(config, epoch, student, gen, int(N), schedule, PredTypes.eps, timestep_type)

                if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                    file_path = config.output_dir + f'/{j}eepoch{epoch}/'
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    student.save_pretrained(file_path)
                accelerator.init_trackers("train_example")
        N = int(N/2)
        teacher.load_state_dict(student.state_dict())

