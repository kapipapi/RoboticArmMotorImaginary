import torch

from gui.booth import Booth
from models.Transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer()
model.load_state_dict(torch.load("gui/transformer.pt", map_location=device))
model.to(device)

Booth(model=model, device=device)
