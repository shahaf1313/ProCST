import torch.nn as nn
import torch.nn.functional as F
import torch
from core.constants import IGNORE_LABEL, NUM_CLASSES
from torch.autograd import Variable
affine_par = True


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)

        return out

class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes, phase):
        self.inplanes = 64
        self.phase = phase
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d( 3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False )
        self.bn1 = nn.BatchNorm2d( 64, affine=affine_par )
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d( kernel_size=3, stride=2, padding=1, ceil_mode=True )  # change

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d( self.inplanes, planes * block.expansion,
                           kernel_size=1, stride=stride, bias=False ),
                nn.BatchNorm2d( planes * block.expansion, affine=affine_par ) )
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append( block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample) )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append( block(self.inplanes, planes, dilation=dilation) )
        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, lbl=None, weight=None, ita=1.5):
        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        if self.training:
            P = F.softmax(x, dim=1)        # [B, 19, H, W]
            logP = F.log_softmax(x, dim=1) # [B, 19, H, W]
            PlogP = P * logP               # [B, 19, H, W]
            ent = -1.0 * PlogP.sum(dim=1)  # [B, 1, H, W]
            ent = ent / 2.9444         # chanage when classes is not 19
            # compute robust entropy
            ent = ent ** 2.0 + 1e-8
            ent = ent ** ita
            self.loss_ent = ent.mean()

            x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
            if lbl is not None:
                self.loss_seg = self.CrossEntropy2d(x, lbl, weight=weight)
            return x, self.loss_seg, self.loss_ent
        x = nn.functional.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

    def get_1x_lr_params_NOscale(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        b.append(self.layer5.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.lr_semseg},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.lr_semseg}]

    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * (  (1-float(i)/args.num_steps) ** (args.power)  )
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  
            
    def CrossEntropy2d(self, predict, target, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))

        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != IGNORE_LABEL)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        target = target.type(torch.long)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)

        return loss    

def Deeplab(num_classes=NUM_CLASSES, phase='train'):
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes, phase)
    return model

