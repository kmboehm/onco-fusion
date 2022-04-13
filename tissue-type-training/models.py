import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch import nn as nn
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg19_bn


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3, type_='vae'):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        if type_ == 'vae':
            self.linear = nn.Linear(512, 2 * z_dim)
        elif type_ == 'cae':
            self.linear = nn.Linear(512, z_dim)
        else:
            raise RuntimeError
        self.type_ = type_

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if self.type_ == 'vae':
            mu = x[:, :self.z_dim]
            logvar = x[:, self.z_dim:]
            return mu, logvar
        elif self.type_ == 'cae':
            return x
        else:
            raise RuntimeError


class ResNet18Dec(nn.Module):

    def __init__(self, activation, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)
        self.activation = activation

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=8)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.conv1(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResNetVAE(nn.Module):

    def __init__(self, z_dim=16, nc=3, activation=torch.tanh, dim=2):
        super().__init__()
        if dim == 2:
            self.encoder = ResNet18Enc(z_dim=z_dim, nc=nc, type_='vae')
            self.decoder = ResNet18Dec(z_dim=z_dim, nc=nc, activation=activation)
        else:
            raise RuntimeError('Must be 2D')

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        if self.training:
            return {'recon_x': x, 'mu': mean, 'logvar': logvar}
        else:
            return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    def encode(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        return z


class ResNetCAE(nn.Module):

    def __init__(self, z_dim=16, nc=3, activation=None):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim, nc=nc, type_='cae')
        self.decoder = ResNet18Dec(z_dim=z_dim, nc=nc, activation=activation)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x

    def encode(self, x):
        return self.encoder(x)


class TissueTileNetProb(torch.nn.Module):
    def __init__(self, model, n_classes):
        self.model = TissueTileNet(model, n_classes, activation=None)

    def forward(self, x):
        y = self.model(x)
        y = torch.nn.Softplus(y)
        y = y + 1
        return y


class TissueTileNet(torch.nn.Module):
    def __init__(self, model, n_classes, activation=None):
        super(TissueTileNet, self).__init__()
        if type(model) in [torchvision.models.resnet.ResNet, TinyCBR]:
            model.fc = torch.nn.Linear(512, n_classes)
        elif type(model) == torchvision.models.squeezenet.SqueezeNet:
            list(model.children())[1][1] = torch.nn.Conv2d(512, n_classes, kernel_size=1, stride=1)
        else:
            raise NotImplementedError
        self.model = model
        self.activation = activation

    def forward(self, x):
        y = self.model(x)

        if self.activation:
            y = self.activation(y)

        return y


def load_tissue_tile_net(checkpoint_path='/gpfs/mskmind_ess/boehmk/histocox/checkpoints/2021-01-19_21.05.24_fold-2_epoch017.torch', activation=None, n_classes=5):
    model = TissueTileNet(resnet18(), n_classes, activation=activation)
    model.load_state_dict(torch.load(
        checkpoint_path,
        map_location='cpu'))
    return model


def get_model(cf):
    if cf.args.model == 'resnet18':
        return resnet18(pretrained=True)
    elif cf.args.model == 'resnet34':
        return resnet34(pretrained=True)
    elif cf.args.model == 'resnet50':
        return resnet50(pretrained=True)
    elif cf.args.model == 'squeezenet':
        return squeezenet1_1(pretrained=True)
    elif cf.args.model == 'cbr-tiny':
        return TinyCBR()
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


class TinyCBR(nn.Module):
    def __init__(self):
        super(TinyCBR, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 128, kernel_size=5, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=5, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 512, kernel_size=5, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
            )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.avg_pool(x)
        x = x.reshape(x.shape[0], x.shape[1])
        x = self.fc(x)
        return x


class SlideAttentionNet(nn.Module):
    def __init__(self, dim, device, dropout=True, return_attention=False, n_heads=1, activation=None, n_classes=1):
        super(SlideAttentionNet, self).__init__()
        self.device = device
        self.return_attention = return_attention
        self.n_heads = n_heads
        self.activation = activation
        self.n_classes = n_classes

        if dropout:
            self.process = nn.Sequential(nn.Linear(dim, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.25),
                                         nn.Linear(512, 512),
                                         nn.ReLU(),
                                         nn.Dropout(0.25))
        else:
            self.process = nn.Sequential(nn.Linear(dim, 512),
                                         nn.ReLU(),
                                         nn.Linear(512, 512),
                                         nn.ReLU())

        self.attentions = nn.ModuleList()
        self.classifiers = nn.ModuleList()
        for _ in range(self.n_heads):
            attention = [nn.Linear(512, 512), nn.Tanh()]
            if dropout:
                attention.append(nn.Dropout(0.25))
            attention.extend([nn.Linear(512, 1),
                             nn.Softmax(1)])
            self.attentions.append(nn.Sequential(*attention))
            # self.classifiers.append(nn.Sequential(nn.Linear(512, 1),
            #                                       nn.Sigmoid()))
            if self.activation:
                self.classifiers.append(nn.Sequential(nn.Linear(512, self.n_classes),
                                                      self.activation))
            else:
                self.classifiers.append(nn.Linear(512, self.n_classes))

    def forward(self, x):
        # print('loaded embeddings: {}'.format(x.shape))
        x = self.process(x)
        # print('processed embeddings: {}'.format(x.shape))

        outputs = []
        for attention, classifier in zip(self.attentions, self.classifiers):
            A = attention(x)
            # print('attention: {}'.format(A.shape))
            # assert torch.allclose(A.sum(1), torch.full_like(A.sum(1), 1))

            M = (A * x).sum(1).squeeze(1)
            # print('M: {}'.format(M.shape))
            # print('attention-gated embeddings: {}'.format(M.shape))

            output = classifier(M).squeeze(-1)
            outputs.append(output)
            # print('S_hat: {}\n'.format(S_hat.shape))

        if self.return_attention:
            return outputs, A
        else:
            return outputs
