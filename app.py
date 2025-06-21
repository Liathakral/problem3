import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ==========================
# Generator Class (same as training)
# ==========================
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

# ==========================
# Constants
# ==========================
NOISE_DIM = 100
LABEL_DIM = 10
IMG_DIM = 28 * 28

# ==========================
# Load the Generator Model
# ==========================
@st.cache_resource
def load_generator():
    model = Generator(NOISE_DIM, LABEL_DIM, IMG_DIM)
    model.load_state_dict(torch.load("generator_problem3.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

generator = load_generator()

# ==========================
# Streamlit UI
# ==========================
st.title("🧠 Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0–9)", list(range(10)))

if st.button("Generate Images"):
    noise = torch.randn(5, NOISE_DIM)
    labels = torch.eye(LABEL_DIM)[[digit]*5]

    with torch.no_grad():
        outputs = generator(noise, labels).view(-1, 28, 28)

    st.subheader(f"Generated Images for Digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        img_array = outputs[i].numpy()
        img_array = (img_array + 1) / 2.0  # Scale from [-1,1] to [0,1]
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array, mode="L")
        cols[i].image(img, caption=f"Sample {i+1}", use_container_width=True)
