import torch
import torch.nn as nn
import torchvision.models
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn


class TissueTileNet(nn.Module):
    def __init__(self, model, n_classes, activation=None):
        super(TissueTileNet, self).__init__()
        if type(model) in [torchvision.models.resnet.ResNet]:
            model.fc = nn.Linear(512, n_classes)
        elif type(model) == torchvision.models.squeezenet.SqueezeNet:
            list(model.children())[1][1] = nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError
        self.model = model
        self.activation = activation

    def forward(self, x):
        y = self.model(x)
        if self.activation:
            y = self.activation(y)

        return y


def get_model(cf):
    if cf.args.model == 'resnet18':
        return resnet18(pretrained=True)
    elif cf.args.model == 'resnet34':
        return resnet34(pretrained=True)
    elif cf.args.model == 'resnet50':
        return resnet50(pretrained=True)
    elif cf.args.model == 'squeezenet':
        return squeezenet1_1(pretrained=True)
    elif cf.args.model == 'vgg19':
        return vgg19_bn(pretrained=True)
    elif cf.args.model == 'tissue-type':
        model = load_tissue_tile_net()
        model = model.model
        for idx, child in enumerate(model.children()):
            if idx in [0, 1, 2, 3, 4, 5, 6, 7]: # 7 is last res block, 9 is fc layer
                for param in child.parameters():
                    param.requires_grad = False
        return model
    else:
        raise RuntimeError("Model type {} unknown".format(cf.model))

def load_tissue_tile_net(checkpoint_path='', activation=None, n_classes=4):
    model = TissueTileNet(resnet18(), n_classes, activation=activation)
    model.load_state_dict(torch.load(
        checkpoint_path,
        map_location='cpu'))
    return model
