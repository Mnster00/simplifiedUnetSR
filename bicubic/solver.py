from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

from VDSR.model import Net
from progress_bar import progress_bar

import pytorch_ssim

class biTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(biTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader

    def build_model(self):
        self.criterion = torch.nn.MSELoss()

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def img_preprocess(self, data, interpolation='bicubic'):
        if interpolation == 'bicubic':
            interpolation = Image.BICUBIC
        elif interpolation == 'bilinear':
            interpolation = Image.BILINEAR
        elif interpolation == 'nearest':
            interpolation = Image.NEAREST

        size = list(data.shape)

        if len(size) == 4:
            target_height = int(size[2] * self.upscale_factor)
            target_width = int(size[3] * self.upscale_factor)
            out_data = torch.FloatTensor(size[0], size[1], target_height, target_width)
            for i, img in enumerate(data):
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Resize((target_width, target_height), interpolation=interpolation),
                                                transforms.ToTensor()])

                out_data[i, :, :, :] = transform(img)
            return out_data
        else:
            target_height = int(size[1] * self.upscale_factor)
            target_width = int(size[2] * self.upscale_factor)
            transform = transforms.Compose([transforms.ToPILImage(),
                                            transforms.Resize((target_width, target_height), interpolation=interpolation),
                                            transforms.ToTensor()])
            return transform(data)



    def test(self):
        #self.model.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data = self.img_preprocess(data)  # resize input image size
                data, target = data.to(self.device), target.to(self.device)
                prediction = data
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                ssim_value = pytorch_ssim.ssim(prediction, target)
                #print(ssim_value)
                avg_ssim += ssim_value
                progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f | SSIM: %.4f' % ((avg_psnr / (batch_num + 1)),avg_ssim / (batch_num + 1)))

        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        self.test()
        #self.scheduler.step(epoch)
