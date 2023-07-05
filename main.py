import torch

from gui.booth import Booth
from models import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer()

# TODO: Rename after training
# model.load_state_dict(torch.load(".model_76_final.pth", map_location=device))
model.to(device)

b = Booth(model=model, device=device)
