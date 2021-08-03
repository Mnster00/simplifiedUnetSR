from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
#from FilterCNN.model import Net
from dataset.data import get_training_set, get_test_set

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

import pytorch_ssim

global upscale_factor
upscale_factor=8

def imshow(img):
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.pause(1)

def img_preprocess(data, interpolation='bicubic'):
    if interpolation == 'bicubic':
        interpolation = Image.BICUBIC
    elif interpolation == 'bilinear':
        interpolation = Image.BILINEAR
    elif interpolation == 'nearest':
        interpolation = Image.NEAREST

    size = list(data.shape)

    if len(size) == 4:
        target_height = int(size[2] * upscale_factor)
        target_width = int(size[3] * upscale_factor)
        out_data = torch.FloatTensor(size[0], size[1], target_height, target_width)
        for i, img in enumerate(data):
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((target_width, target_height), interpolation=interpolation),
                                            transforms.ToTensor()])

            out_data[i, :, :, :] = transform(img)
        return out_data
    else:
        target_height = int(size[1] * upscale_factor)
        target_width = int(size[2] * upscale_factor)
        transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((target_width, target_height), interpolation=interpolation),
                                        transforms.ToTensor()])
        return transform(data)



CUDA = torch.cuda.is_available()
device = torch.device('cpu')
test_set = get_test_set(upscale_factor)
testing_loader=testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)




for batch_num, (data, target) in enumerate(testing_loader):


    data, target = data.to(device), target.to(device)
    #target = img_preprocess(target)
    model_out_path = "icdar2003srcnn_8.pth"
    net1=torch.load(model_out_path)
    net1.to(device)
    net1.eval()

    prediction = net1(data) 
    del net1 




    img_PIL_1,Cb,Cr= transforms.ToPILImage()(data.squeeze(0)).convert('YCbCr').split()
    data2 = transforms.ToTensor()(img_PIL_1).unsqueeze(0)

    model_out_path = "icdar2003sub_8.pth"
    net2=torch.load(model_out_path)
    net2.to(device)
    net2.eval()

    prediction2 = net2(data2)
    del net2

    out_img_y = prediction2.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = Cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = Cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')



    model_out_path = "icdar2003dbpn_8.pth"
    net3=torch.load(model_out_path)
    net3.to(device)
    net3.eval()

    prediction3 = net3(data)
    del net3




    model_out_path = "icdar2003drcn_8.pth"
    net4=torch.load(model_out_path)
    net4.to(device)
    net4.eval()

    _,prediction4 = net4(data)
    del net4


    model_out_path = "icdar2003vdsr_8.pth"
    net5=torch.load(model_out_path)
    net5.to(device)
    net5.eval()

    prediction5 = net5(data)
    del net5


    model_out_path = "icdar2003unet_8_l2.pth"
    net6=torch.load(model_out_path)
    net6.to(device)
    net6.eval()

    prediction6 = net6(data)
    del net6



    model_out_path = "icdar2003unet_8_l3.pth"
    net7=torch.load(model_out_path)
    net7.to(device)
    net7.eval()

    prediction7 = net7(data)
    del net7

    source=img_preprocess(data,'bicubic')
    source=source.squeeze(0).squeeze(0)
    target=target.squeeze(0).squeeze(0)
    prediction=prediction.squeeze(0).squeeze(0)

    prediction3=prediction3.squeeze(0).squeeze(0)
    prediction4=prediction4.squeeze(0).squeeze(0)
    prediction5=prediction5.squeeze(0).squeeze(0)
    prediction6=prediction6.squeeze(0).squeeze(0)
    prediction7=prediction7.squeeze(0).squeeze(0)


    npimg_1 = source.detach().numpy()
    npimg0 = target.detach().numpy()
    npimg1 = prediction.detach().numpy()

    npimg3 = prediction3.detach().numpy()
    npimg4 = prediction4.detach().numpy()
    npimg5 = prediction5.detach().numpy()
    npimg6 = prediction6.detach().numpy()
    npimg7 = prediction7.detach().numpy()

    plt.figure()
    plt.subplot(2,5,1)
    plt.imshow(np.transpose(npimg0,(1,2,0)))#source
    plt.subplot(2,5,2)
    plt.imshow(np.transpose(npimg_1,(1,2,0)))#source
    plt.subplot(2,5,3)
    plt.imshow(np.transpose(npimg1,(1,2,0)))#srcnn
    plt.subplot(2,5,4)
    plt.imshow(np.array(out_img))#sub!!!
    plt.subplot(2,5,5)
    plt.imshow(np.transpose(npimg3,(1,2,0)))#dbpn
    plt.subplot(2,5,6)
    plt.imshow(np.transpose(npimg0,(1,2,0)))#drcn
    plt.subplot(2,5,7)
    plt.imshow(np.transpose(npimg4,(1,2,0)))#drcn
    plt.subplot(2,5,8)
    plt.imshow(np.transpose(npimg5,(1,2,0)))#vdsr
    plt.subplot(2,5,9)
    plt.imshow(np.transpose(npimg6,(1,2,0)))#unet
    plt.subplot(2,5,10)
    plt.imshow(np.transpose(npimg7,(1,2,0)))#unet

    plt.xticks([])
    plt.yticks([])
    frame = plt.gca()
    # y 轴不可见
    frame.axes.get_yaxis().set_visible(False)
    # x 轴不可见
    frame.axes.get_xaxis().set_visible(False)
    plt.axis('off')
    plt.show()
    plt.pause(1)





