from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor,Compose, CenterCrop, Resize
from torchvision import transforms

import numpy as np

def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)



# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default='E:/torchtest/SR/super-resolution-master/dataset/BSDS300/images/val/3096.jpg', help='input image to use')
parser.add_argument('--model', type=str, default='model_path.pth', help='model file to use')
parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
args = parser.parse_args()
print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
img = Image.open(args.input).convert('RGB')
crop_size = calculate_valid_crop_size(256, 4)

transform1=transforms.Compose([
        CenterCrop(crop_size),
        ToTensor(),
        ]
)

img = transform1(img)
print(img.shape)



# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cpu')#('cuda' if GPU_IN_USE else 'cpu')
model = torch.load(args.model, map_location=lambda storage, loc: storage)
model = model.to(device)
data = img.unsqueeze(0)
data = data.to(device)
print(data.shape)
if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================
out = model(data)
out = out.cpu()
print(out.data[0].shape)
out = out.data[0]
#print(out)
print(out.shape)

#out_img_y = out.data[0].numpy()

#out_img_y *= 255.0
#out_img_y = out_img_y.clip(0, 255)
#unloader = transforms.ToPILImage()
#out_img = unloader(np.uint8(out))
img_PIL = transforms.ToPILImage()(out).convert('RGB')
#img_PIL.show() 
#print(out)
#out_img = Image.fromarray(np.uint8(out), mode='L')

#out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
#out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
#out_img = Image.merge('RGB', out).convert('RGB')

img_PIL.save(args.output)
print('output image saved to ', args.output)