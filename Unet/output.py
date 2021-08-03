from __future__ import print_function

from math import log10

import torch
import torch.backends.cudnn as cudnn

#from FilterCNN.model import Net
from Unet.Umodel import UNet8
from Unet.Umodel import UNet4
from Unet.Umodel import UNet2
from dataset.data import get_training_set, get_test_set

import pytorch_ssim

self.upscale_factor=2
self.CUDA = torch.cuda.is_available()
self.device = torch.device('cuda' if self.CUDA else 'cpu')
test_set = get_test_set(args.upscale_factor)
self.testing_loader=testing_data_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)



     
if self.upscale_factor==2:
    self.model=UNet2(3,3).to(self.device)
if self.upscale_factor==4:
    self.model=UNet4(3,3).to(self.device)
if self.upscale_factor==8:
    self.model=UNet8(3,3).to(self.device)

model_out_path = "icdar2003unet_2.pth"
torch.load(self.model, model_out_path)
    
print(net)

'''
    def test(self):
        self.model.eval()
        avg_psnr = 0
        avg_ssim = 0
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                #data=img2filter(data)#!!!
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
                self.save_model()
'''