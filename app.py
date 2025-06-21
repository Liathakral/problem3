import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import gradio as gr

# Generator class
class Generator(nn.Module):
    def __init__(self, noise_dim, label_dim, image_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + label_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, image_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], dim=1)
        return self.model(x)

# Constants
NOISE_DIM = 100
LABEL_DIM = 10
IMG_DIM = 28 * 28

# Load the trained generator model
generator = Generator(NOISE_DIM, LABEL_DIM, IMG_DIM)
generator.load_state_dict(torch.load("generator_problem3.pth", map_location=torch.device('cpu')))
generator.eval()

def generate_images(digit):
    noise = torch.randn(5, NOISE_DIM)
    labels = torch.eye(LABEL_DIM)[[digit] * 5]
    with torch.no_grad():
        outputs = generator(noise, labels).view(-1, 28, 28)

    images = []
    for output in outputs:
        img_array = output.numpy()
        img_array = (img_array + 1) / 2.0
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        images.append(img)
    return images

# Gradio UI
demo = gr.Interface(
    fn=generate_images,
    inputs=gr.Slider(0, 9, step=1, label="Digit"),
    outputs=gr.Gallery(label="Generated Images").style(grid=5),
    title="🧠 Handwritten Digit Generator"
)

if __name__ == "__main__":
    demo.launch()
