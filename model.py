import torch.nn as nn

class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, bias= False, stride = 1, padding =1, pool = False, dropout = 0):
        super(ConvLayer,self).__init__()

        layers =  []
        layers.append(
            nn.Conv2d(input_channels, output_channels, 3, bias=bias, stride= stride, padding= padding, padding_mode='replicate')
        )

        if pool:
            layers.append(
                nn.MaxPool2d(2,2)
            )
        layers.append(
            nn.BatchNorm2d(output_channels)
        )
        layers.append(
            nn.ReLU()
        )

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        self.all_layers = nn.Sequential(*layers)

class CustomLayer(nn.Module):
    def __init__(self, input_channels, output_channels, pool = True, reps = 2, dropout = 0):
        super(CustomLayer, self).__init__()

        self.pool_layer = ConvLayer(input_channels, output_channels, pool=pool, dropout= dropout)
        self.res_layer = None

        if reps > 0:
            layers = []
            for i in range(0, reps):
                layers.append(
                    ConvLayer(output_channels, output_channels, pool= False, dropout=dropout)
                )
            self.res_layer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool_layer(x)
        

        
