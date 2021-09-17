import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
# from tensorboardX import SummaryWriter



class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
 
    def forward(self, x):
        return self.double_conv(x)
        
        
        
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
 
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
 
    def forward(self, x):
        return self.maxpool_conv(x)




class Up(nn.Module):
    """Upscaling then double conv"""
 
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
 
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
 
        self.conv = DoubleConv(in_channels, out_channels)
 
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
 
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
 
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
        
        
        
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.conv(x)

class CompressConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CompressConv, self).__init__()
        self.compress = nn.Conv2d(in_channels, out_channels, kernel_size=1)
 
    def forward(self, x):
        return self.compress(x)
        
        
        
class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        pretrained_model = resnet.__dict__['resnet{}'.format(18)](pretrained=True)
 
        # rgb pipeline
        self.inc_rgb = DoubleConv(3, 64)
        self.resnet1_rgb = pretrained_model._modules['layer1'] # 64 as input
        self.resnet2_rgb = pretrained_model._modules['layer2'] # 128 as output
        # self.down1_rgb = Down(64, 128)
        self.down2_rgb = Down(128, 256)
        self.down3_rgb = Down(256, 512)
        # thermal pipeline
        self.inc_t = DoubleConv(1, 64)
        self.resnet1_t = pretrained_model._modules['layer1'] # 64 as input
        self.resnet2_t = pretrained_model._modules['layer2'] # 128 as output
        # self.down1 = Down(64, 128)
        self.down2_t = Down(128, 256)
        self.down3_t = Down(256, 512)
        # fusion seg model
        self.down4 = Down(512, 512)
        self.compress1 = CompressConv(128,64)
        self.compress2 = CompressConv(256,128)
        self.compress3 = CompressConv(512,256)
        self.compress4 = CompressConv(1024,512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
 
    def forward(self, x):
        # split two branches: rgb, t

        # rgb feature extraction
        x_rgb = x[:,0:3,:,:]
        x1_rgb = self.inc_rgb(x_rgb)
        # rgb backbone
        x1_rgb = self.resnet1_rgb(x1_rgb)
        x2_rgb = self.resnet2_rgb(x1_rgb)
        # x2_rgb = self.down1_rgb(x1_rgb)
        x3_rgb = self.down2_rgb(x2_rgb)
        x4_rgb = self.down3_rgb(x3_rgb)

        # thermal feature extraction
        x_t = x[:,3,:,:].unsqueeze(1)
        x1_t = self.inc_t(x_t)
        # thermal backbone
        x1_t = self.resnet1_t(x1_t)
        x2_t = self.resnet2_t(x1_t)
        # x2_t = self.down1_t(x1_t)
        x3_t = self.down2_t(x2_t)
        x4_t = self.down3_t(x3_t)

        # fusion by concat
        x1 = torch.cat((x1_rgb,x1_t),1)
        x2 = torch.cat((x2_rgb,x2_t),1)
        x3 = torch.cat((x3_rgb,x3_t),1)
        x4 = torch.cat((x4_rgb,x4_t),1)
        x1 = self.compress1(x1)
        x2 = self.compress2(x2)
        x3 = self.compress3(x3)
        x4 = self.compress4(x4)
        # post-fusion processing
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits