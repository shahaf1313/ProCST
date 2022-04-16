import torch
import torch.nn as nn
import torchvision


def gram(x):
    b, c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2))
    return g.div(h * w)


def calc_tv_loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


class StyleTransferLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.to(opt.device)
        self.mse_criterion = torch.nn.MSELoss(reduction='mean')
        self.feature_extractor_model = torchvision.models.__dict__['vgg16'](pretrained=True).features.to(opt.device)
        self.content_layers = opt.content_layers
        self.style_layers = opt.style_layers
        self.content_weight = opt.content_weight
        self.style_weight = opt.style_weight
        self.tv_weight = opt.tv_weight

    def forward(self, fake_image, fake_content_features, fake_style_features, source_content_features, target_style_features):
        content_loss = self.calc_content_loss(fake_content_features, source_content_features)
        style_loss = self.calc_gram_loss(fake_style_features, target_style_features)
        tv_loss = calc_tv_loss(fake_image)
        total_style_loss = content_loss * self.content_weight + style_loss * self.style_weight + tv_loss * self.tv_weight
        style_losses = {'ContentLoss': content_loss.item(),
                        'StyleLoss': style_loss.item(),
                        'TVLoss': tv_loss.item(),
                        'TotalStyleLoss': total_style_loss.item()}
        return total_style_loss, style_losses

    def calc_content_loss(self, features, targets, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)

        content_loss = 0
        for f, t, w in zip(features, targets, weights):
            content_loss += self.mse_criterion(f, t) * w

        return content_loss

    def calc_gram_loss(self, features, targets, weights=None):
        if weights is None:
            weights = [1 / len(features)] * len(features)

        gram_loss = 0
        for f, t, w in zip(features, targets, weights):
            gram_loss += self.mse_criterion(gram(f), gram(t)) * w
        return gram_loss

    def extract_features(self, image):
        content_features, style_features = [], []
        x = image
        for index, layer in enumerate(self.feature_extractor_model):
            x = layer(x)
            if index in self.content_layers:
                content_features.append(x)
            if index in self.style_layers:
                style_features.append(x)
        return content_features, style_features
