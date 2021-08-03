import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np



class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel1 = [[-1., -2., -1.],
                  [0., 0., 0.],
                  [1., 2., 1.]]
        kernel1 = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        self.weight1 = nn.Parameter(data=kernel1, requires_grad=False)

        kernel2 = [[-1., 0., 1.],
                  [-2., 0., 2.],
                  [-1., 0., 1.]]
        kernel2 = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        self.weight2 = nn.Parameter(data=kernel2, requires_grad=False)

    def forward(self, x, y):

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight1, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight1, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight1, padding=2)

        x = torch.cat([x1, x2, x3], dim=1)

        y1 = y[:, 0]
        y2 = y[:, 1]
        y3 = y[:, 2]
        y1 = F.conv2d(y1.unsqueeze(1), self.weight1, padding=2)
        y2 = F.conv2d(y2.unsqueeze(1), self.weight1, padding=2)
        y3 = F.conv2d(y3.unsqueeze(1), self.weight1, padding=2)

        y = torch.cat([y1, y2, y3], dim=1)
        loss1=torch.mean(torch.mean((x-y)**2))/100.0

        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight2, padding=2)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight2, padding=2)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight2, padding=2)

        x = torch.cat([x1, x2, x3], dim=1)

        y1 = y[:, 0]
        y2 = y[:, 1]
        y3 = y[:, 2]
        y1 = F.conv2d(y1.unsqueeze(1), self.weight2, padding=2)
        y2 = F.conv2d(y2.unsqueeze(1), self.weight2, padding=2)
        y3 = F.conv2d(y3.unsqueeze(1), self.weight2, padding=2)

        y = torch.cat([y1, y2, y3], dim=1)
        loss2=torch.mean(torch.mean((x-y)**2))/10000.0

        loss=torch.sqrt(loss1*loss1+loss2*loss2)*2
        #print(loss)
        #loss=(torch.log(loss))/100.0
        #print(loss)
        return loss