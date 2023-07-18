
import torch.nn as nn
import torchinfo

class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias=False, stride=1, padding=1, pool=False, dropout=0):
        super(ConvLayer, self).__init__()

        layers = list()
        layers.append(
            nn.Conv2d(input_channels, output_channels, 3, bias=bias, stride=stride, padding=padding, padding_mode='replicate')
        )
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        layers.append(nn.BatchNorm2d(output_channels))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.all_layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.all_layers(x)


class CustomLayer(nn.Module):
    def __init__(self, input_channels, output_channels, pool=True, res=2, dropout=0):
        super(CustomLayer, self).__init__()

        self.conv_layer = ConvLayer(input_channels, output_channels, pool=pool, dropout=dropout)
        self.res_layer = None
        if res > 0:
            layers = list()
            for i in range(0, res):
                layers.append(
                    ConvLayer(output_channels, output_channels, pool=False, dropout=dropout)
                )
            self.res_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layer(x)
        if self.res_layer is not None:
            x_ = x
            x = self.res_layer(x)
            x = x + x_
        return x


class CustomResNet(nn.Module):
    def __init__(self, dropout=0):
        super(CustomResNet, self).__init__()

        self.all_layers = nn.Sequential(
            CustomLayer(input_channels= 3, output_channels= 64, pool=False, res=0, dropout=dropout),
            CustomLayer(input_channels= 64, output_channels= 128, pool=True, res=2, dropout=dropout),
            CustomLayer(input_channels= 128, output_channels= 256, pool=True, res=0, dropout=dropout),
            CustomLayer(input_channels= 256, output_channels= 512, pool=True, res=2, dropout=dropout),
            nn.MaxPool2d(4,4),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.all_layers(x)
    
    # def summary(self, input_size=None, depth=10):
    #     return torchinfo.summary(self, input_size=input_size, depth=depth,
    #                              col_names=["input_size", "output_size", "num_params", "params_percent"])

