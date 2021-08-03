from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from SubPixelCNN.model import Net
from progress_bar import progress_bar

import pytorch_ssim

class SubPixelTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(SubPixelTrainer, self).__init__()
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
        self.model = Net(upscale_factor=self.upscale_factor).to(self.device)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        print('# model parameters:', sum(param.numel() for param in self.model.parameters()))
        
        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

    def save(self):
        model_out_path = "sub.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            #print(data.shape)
            img_PIL_1,Cb,Cr= transforms.ToPILImage()(data.squeeze(0)).convert('YCbCr').split()
            data = transforms.ToTensor()(img_PIL_1).unsqueeze(0)
            img_PIL_2,Cb,Cr= transforms.ToPILImage()(target.squeeze(0)).convert('YCbCr').split()
            target = transforms.ToTensor()(img_PIL_2).unsqueeze(0)
            #print(data.shape)
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                img_PIL_1,Cb,Cr= transforms.ToPILImage()(data.squeeze(0)).convert('YCbCr').split()
                data = transforms.ToTensor()(img_PIL_1).unsqueeze(0)
                img_PIL_2,Cb,Cr= transforms.ToPILImage()(target.squeeze(0)).convert('YCbCr').split()
                target = transforms.ToTensor()(img_PIL_2).unsqueeze(0)
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
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
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
            if epoch == self.nEpochs:
                self.save()
