import matplotlib.pyplot as plt
import torch
from functools import partial
from matplotlib.animation import FuncAnimation

class PlotAnimation:
    def __init__(self):
        self.fig = plt.figure(figsize=(15, 15))
        self.ani = 0
        self.imgs = []
        self.index = 0

    def animate(self, i):
        plt.imshow(self.imgs[i])

    def start_animation(self):
        self.ani = FuncAnimation(self.fig, partial(self.animate), interval=10, frames=len(self.imgs))
        plt.show()
    def add_img(self, image):
        self.imgs.append(image)
        self.index += 1

    def add_tensor(self, t):
        image = (t / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy().astype(float)[0]
        self.add_img(image)


