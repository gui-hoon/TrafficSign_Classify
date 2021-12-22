import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from torch2trt import torch2trt

class CustomConvNet(nn.Module):
    def __init__(self):
        super(CustomConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, 10)
        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))

class_name = ['70','None', '120', '50', '100', '40', '90', '30', '80', '60']

transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])

img = Image.open("./30.14.png")
img = img.convert("RGB")
img = transform(img)
img =img.view(1, 3, 128, 128).cuda()

model = torch.load("./classification_model.pth")
model.eval().cuda()

# create example data
x = torch.ones((1, 3, 128, 128)).cuda()
# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x], fp32_mode = True)

# torch model
# with torch.no_grad():

#     outputs = model(img)
#     idx = torch.argmax(outputs)
#     print(class_name[idx])

# trt model
with torch.no_grad():

    outputs = model_trt(img)
    idx = torch.argmax(outputs)
    print(class_name[idx])

