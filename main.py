import torch

from gui.booth import Booth
from models.EEGInception import EEGInception

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EEGInception()

model.load_state_dict(torch.load("gui/inception.pt", map_location=device))
model.to(device)

b = Booth(model=model, device=device)
