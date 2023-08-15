import torch
from model import MultiTaskModel
import torchvision.models as models

model = MultiTaskModel("resnet34")


checkpoint = torch.load("resnet34-333f7ec4.pth", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint, strict=False)
img = torch.randn(1,3,224,224)
save_path = "./bak.onnx"
torch.onnx.export(model, img, save_path, input_names=["input"], opset_version=10)
