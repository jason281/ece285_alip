import torch
import torch.nn as nn
import torch.nn.fuctioncal as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)\n",
        self.pool3 = nn.MaxPool2d(2,2)\n",

        self.conv7 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv8 = nn.Conv2d(128, 128, 3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv9 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv10 = nn.Conv2d(256, 256, 3, padding=1)
        self.batchnorm5 = nn.BatchNorm2d(256)

        self.convtran1 = nn.ConvTranspose2d(256,128,2,stride=2)
        
        self.conv11 = nn.Conv2d(256,128,3, padding=1)
        self.conv12 = nn.Conv2d(128, 128,3, padding=1)
        self.batchnorm6 = nn.BatchNorm2d(128)

        self.convtran2 = nn.ConvTranspose2d(128,64,2,stride=2)

        self.conv13 = nn.Conv2d(128,64,3, padding=1)
        self.conv14 = nn.Conv2d(64,64,3, padding=1)
        self.batchnorm7 = nn.BatchNorm2d(64)

        self.convtran3 = nn.ConvTranspose2d(64,32,2,stride=2)

        self.conv15 = nn.Conv2d(64,32,3, padding=1)
        self.conv16 = nn.Conv2d(32,32,3, padding=1)
        self.batchnorm8 = nn.BatchNorm2d(32)

        self.convtran4 = nn.ConvTranspose2d(32,16,2,stride=2)

        self.conv17 = nn.Conv2d(32,16,3, padding=1)
        self.conv18 = nn.Conv2d(16, 16,3, padding=1)
        self.batchnorm9 = nn.BatchNorm2d(16)
              
        self.conv19 = nn.Conv2d(16,3,1)
        
    def forward(self, x):
        c1 = F.relu(self.batchnorm1(self.conv1(x)))
        c1 = F.relu(self.batchnorm1(self.conv2(c1)))
        p1 = self.pool1(c1)

        c2 = F.relu(self.batchnorm2(self.conv3(p1)))
        c2 = F.relu(self.batchnorm2(self.conv4(c2)))
        p2 = self.pool2(c2)

        c3 = F.relu(self.batchnorm3(self.conv5(p2)))
        c3 = F.relu(self.batchnorm3(self.conv6(c3)))
        p3 = self.pool3(c3)

        c4 = F.relu(self.batchnorm4(self.conv7(p3)))
        c4 = F.relu(self.batchnorm4(self.conv8(c4)))
        p4 = self.pool4(c4)

        c5 = F.relu(self.batchnorm5(self.conv9(p4)))
        c5 = F.relu(self.batchnorm5(self.conv10(c5)))

        u6 = self.convtran1(c5)
        u6 = torch.cat((u6,c4),dim=1)

        c6 = F.relu(self.batchnorm6(self.conv11(u6)))
        c6 = F.relu(self.batchnorm6(self.conv12(c6)))

        u7 = self.convtran2(c6)
        u7 = torch.cat((u7,c3),dim=1)

        c7 = F.relu(self.batchnorm7(self.conv13(u7)))
        c7 = F.relu(self.batchnorm7(self.conv14(c7)))

        u8 = self.convtran3(c7)
        u8 = torch.cat((u8,c2),dim=1)

        c8 = F.relu(self.batchnorm8(self.conv15(u8)))
        c8 = F.relu(self.batchnorm8(self.conv16(c8)))

        u9 = self.convtran4(c8)
        u9 = torch.cat((u9,c1),dim=1)

        c9 = F.relu(self.batchnorm9(self.conv17(u9)))
        c9 = F.relu(self.batchnorm9(self.conv18(c9)))

        out = F.sigmoid(self.conv19(c9))
        
        return out
    
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)