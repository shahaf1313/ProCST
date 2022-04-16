import torch
import torch.nn as nn

# Blocks:
class ConvBlockProCST(nn.Module):
    def __init__(self, in_channel, out_channel, ker_size, im_per_gpu, groups_num, padd=1, stride=1):
        super(ConvBlockProCST, self).__init__()
        # Normalization:
        if im_per_gpu >= 16:
            self.norm = nn.BatchNorm2d(in_channel)
        elif im_per_gpu < 16 and in_channel % groups_num == 0:
            self.norm = nn.GroupNorm(num_groups=groups_num, num_channels=in_channel, affine=True)
        else: #don't normalize only in the head module, where you have only 3 channels..
            self.norm  = None
        # Activation:
        self.actvn = nn.LeakyReLU(0.2)
        # Convolution:
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=stride, padding=padd)

    def forward(self, x):
        if self.norm==None: #Don't use norm layer:
            z = self.actvn(self.conv(x))
        else:
            z = self.conv(self.actvn(self.norm(x)))
        return z

# Generator:
class ProCSTGenerator(nn.Module):
    def __init__(self, opt):
        super(ProCSTGenerator, self).__init__()
        self.images_per_gpu = opt.images_per_gpu[opt.curr_scale]
        self.is_initial_scale = opt.curr_scale == 0
        self.layers_in_generator = opt.body_layers if opt.curr_scale < opt.num_scales - 1 else opt.body_layers+2
        self.head = ConvBlockProCST((2 - self.is_initial_scale) * opt.nc_im, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
        self.body = nn.Sequential()
        for i in range(self.layers_in_generator-2):
            block = ConvBlockProCST(opt.base_channels, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(ConvBlockProCST(opt.base_channels, opt.nc_im, opt.ker_size, self.images_per_gpu, opt.groups_num),
                                      nn.Tanh())
    def forward(self, curr_scale, prev_scale):
        if self.is_initial_scale:
            z = curr_scale
        else:
            z = torch.cat((curr_scale, prev_scale), 1)
        z = self.head(z)
        z = self.body(z)
        z = self.tail(z)
        return z

# Discriminator:
class ProCSTDiscriminator(nn.Module):
    def __init__(self, opt):
        super(ProCSTDiscriminator, self).__init__()
        self.images_per_gpu = opt.images_per_gpu[opt.curr_scale]
        self.layers_in_discriminator = opt.body_layers if opt.curr_scale < opt.num_scales - 1 else opt.body_layers+2
        self.head = ConvBlockProCST(opt.nc_im, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
        self.body = nn.Sequential()
        for i in range(self.layers_in_discriminator-2):
            block = ConvBlockProCST(opt.base_channels, opt.base_channels, opt.ker_size, self.images_per_gpu, opt.groups_num)
            self.body.add_module('block%d'%(i+1),block)
        self.tail = nn.Sequential(ConvBlockProCST(opt.base_channels, 1, opt.ker_size, self.images_per_gpu, opt.groups_num),
                                  nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

# Miscellaneous:
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        if m.affine == True:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

