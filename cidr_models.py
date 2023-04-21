from torch import nn

class ConvBNRelu( nn.Sequential ):
    """
    Convolutional layer with batchnorm and clipping ReLU
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNRelu, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual( nn.Module ):
    """
    Bottleneck Block
    """
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual,self).__init__()
        self.stride = stride
    
        assert stride in [1,2]

        hidden_dim = int(round(expand_ratio*inp))
        
        # Residual connection only if channels are preserved (inp == oup).
        # and the image dimension is preserved (stride == 1)
        
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []

        if expand_ratio != 1:
            layers.append(ConvBNRelu(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # Depthwise Convolution/Fully-grouped Convolution
            ConvBNRelu(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            
            # Pointwise Linear Convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self,x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class CIDR_MobileNetv2( nn.Module ):
    """
    CIDR Implementation of MBV2
    """
    def __init__(self, num_classes=10, width_mult=1.0):
        super(CIDR_MobileNetv2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        block_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # first layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNRelu(3, input_channel, stride=2)]

        self.last_channel = int(last_channel * max(1.0, width_mult))
        
        # blocks
        for t, c, n, s in block_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append( block(input_channel, output_channel, stride, expand_ratio=t) )
                input_channel = output_channel
        
        # last layer
        features.append( ConvBNRelu(input_channel, self.last_channel, kernel_size=1) )
        
        self.features = nn.Sequential(*features)

        self.projector = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(self.last_channel, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        # what's the point of taking this mean?
        x = x.mean(-1).mean(-1)
        x = self.projector(x)
        return x
