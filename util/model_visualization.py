import torch
import torchvision
from tensorboardX import SummaryWriter


dummy_input = torch.Tensor(1, 3, 224, 224)

with SummaryWriter(comment='resnet18') as w:
    model = torchvision.models.resnet18()
    w.add_graph(model, (dummy_input, ))


