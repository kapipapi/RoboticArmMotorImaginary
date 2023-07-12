import torch

from gui.booth import Booth
from models.Transformer import Transformer

model = Transformer()
model.load_state_dict(torch.load("gui/transformer.pt"))

Booth(model=model)
