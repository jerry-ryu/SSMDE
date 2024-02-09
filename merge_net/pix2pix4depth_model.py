import torch
import torch.nn as nn
import networks


class Pix2Pix4DepthModel(nn.Module):
    
    def __init__(self):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(Pix2Pix4DepthModel, self).__init__()
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(2, 1, 64, 'unet_1024', 'none',False, 'normal', 0.02)
        
    def normalize(self, img):
        img = img * 2
        img = img - 1
        return img

    def normalize01(self, img):
        return (img - torch.min(img)) / (torch.max(img)-torch.min(img))
        
    def preprocessing(self,input):
        input = self.normalize01(input)
        input = self.normalize(input)
        return input        

    def forward(self, high_input, low_input, high_height, high_width):
        self.outer = high_input["disp", 0]
        self.outer = torch.nn.functional.interpolate(self.outer,(high_height,high_width),mode='bilinear',align_corners=False)
        # self.outer = self.preprocessing(self.outer)

        self.inner = low_input["disp", 0]
        self.inner = torch.nn.functional.interpolate(self.inner,(high_height,high_width),mode='bilinear',align_corners=False)
        # self.inner = self.preprocessing(self.inner)

        self.real_A = torch.cat((self.outer, self.inner), 1)
        self.fake_B = self.netG(self.real_A)  # G(A)
        return self.fake_B

if __name__=='__main__':
    unet = Pix2Pix4DepthModel()
    print(sum(p.numel() for p in unet.parameters()))
