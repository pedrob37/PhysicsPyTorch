import torch.nn as nn
import torch.nn.functional as F
from torch import cat
from base import BaseModel


class nnUnetConvBlock(nn.Module):
    """
    The basic convolution building block of nnUnet.
    """

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        # Activation function
        self.nonlin = nn.LeakyReLU
        # Dropout type (Also have 2d, 3d)
        self.dropout = nn.Dropout
        if self.dropout is None:
            print('Initialising model without dropout.')
        # Normalisation type
        self.norm = None
        # Convolution type (3D for all experiments)
        self.conv = nn.Conv3d

        # Unknown parameter
        self.nonlin_args = None # {'negative_slope': 1e-2, 'inplace': True}

        # Dropout probability
        self.dropout_args = {'p': 0.5, 'inplace': True}
        #self.norm_args = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        # Normalisation arguments: If relevant
        self.norm_args = None # {'eps': 1e-5, 'affine': True}

        # Convolution arguments
        self.conv_args = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv = self.conv(self.input_channels, self.output_channels, **self.conv_args)
        if self.dropout is not None and self.dropout_args['p'] is not None and self.dropout_args['p'] > 0:
            self.dropout = self.dropout(**self.dropout_args)
        else:
            self.dropout = None
        #self.norm = self.norm(num_features=self.output_channels, **self.norm_args)
        #print('Output channels: {} BatchNorm groups: {}'.format(self.output_channels, self.output_channels//4))
        self.norm = self.norm(num_channels=self.output_channels, num_groups=self.output_channels//2, **self.norm_args)
        self.nonlin = self.nonlin(**self.nonlin_args)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.norm(x)
        return self.nonlin(x)

class nnUnetConvBlockStack(nn.Module):
    """
    Concatenates multiple nnUnetConvBlocks.
    """
    def __init__(self, input_channels, output_channels, num_blocks):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stack = nn.Sequential(*([nnUnetConvBlock(input_channels, output_channels)]
                                   +[nnUnetConvBlock(output_channels, output_channels) for _ in range(num_blocks-1)]))


    def forward(self, x):
        return self.stack(x)

class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class nnUnet(BaseModel):
    """
    Stripped-down implementation of nnUnet.
     Notes:
         Based on code in nnUnet repo https://github.com/MIC-DKFZ/nnUNet.
    """
    def __init__(self, input_channels, base_num_channels=30, num_pool=3, num_classes=151, uncertainty_classes=0):
        super().__init__()
        self.upsample_mode = 'trilinear'
        self.pool = nn.MaxPool3d
        self.uncertainty_classes = uncertainty_classes
        #self.transposed_conv = nn.ConvTranspose3d

        self.downsample_path_convs = []
        self.downsample_path_pooling = []
        self.upsample_path_convs = []
        self.upsample_path_upsampling = []
        if self.uncertainty_classes > 0:
            self.upsample_path_convs_uncertainty = []
            self.upsample_path_upsampling_uncertainy = []
        self.num_classes = num_classes


        # build the downsampling pathway
        # initialise channel numbers for first level
        #input_channels = input_channels # specified as argument
        output_channels = base_num_channels
        for level in range(num_pool):
            # Add two convolution blocks
            self.downsample_path_convs.append(nnUnetConvBlockStack(input_channels, output_channels, 2))

            # Add pooling
            self.downsample_path_pooling.append(self.pool([2,2,2]))

            # Calculate input/output channels for next level
            input_channels = output_channels
            output_channels *= 2

        # now the 'bottleneck'
        final_num_channels = self.downsample_path_convs[-1].output_channels
        self.downsample_path_convs.append(nn.Sequential(nnUnetConvBlockStack(input_channels, output_channels, 1),
                                                        nnUnetConvBlockStack(output_channels, final_num_channels,1)))

        # now build the upsampling pathway
        for level in range(num_pool):
            channels_from_down = final_num_channels
            channels_from_skip = self.downsample_path_convs[-(2 + level)].output_channels
            channels_after_upsampling_and_concat = channels_from_skip * 2

            if level != num_pool-1:
                final_num_channels = self.downsample_path_convs[-(3+level)].output_channels
            else:
                final_num_channels = channels_from_skip

            self.upsample_path_upsampling.append(Upsample(scale_factor=[2,2,2], mode=self.upsample_mode))
            #self.upsample_path_upsampling.append(nn.ConvTranspose3d(channels_from_skip, channels_from_skip, 3, stride=2, output_padding=1))

            # Add two convs
            self.upsample_path_convs.append(nn.Sequential(nnUnetConvBlockStack(channels_after_upsampling_and_concat, channels_from_skip,1),
                                                          nnUnetConvBlockStack(channels_from_skip, final_num_channels,1)))
            if self.uncertainty_classes > 0:
                self.upsample_path_upsampling_uncertainy.append(Upsample(scale_factor=[2, 2, 2], mode=self.upsample_mode))
                self.upsample_path_convs_uncertainty.append( nn.Sequential(nnUnetConvBlockStack(channels_after_upsampling_and_concat, channels_from_skip, 1),
                                  nnUnetConvBlockStack(channels_from_skip, final_num_channels, 1)))

        if self.uncertainty_classes > 0:
            self.uncertainty_output = nn.Sequential(nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels, self.upsample_path_convs[-1][-1].output_channels//2, 1, 1, 0, 1, 1, bias=True),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels//2,self.upsample_path_convs[-1][-1].output_channels // 3, 1, 1, 0, 1, 1, bias=True),
                                                    nn.LeakyReLU(),
                                                    nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels//3,  uncertainty_classes, 1, 1, 0, 1, 1, bias=True))
            self.segmentation_output = nn.Sequential(nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels, self.upsample_path_convs[-1][-1].output_channels * 2, 1, 1, 0, 1, 1, bias=True),
                                                     nn.LeakyReLU(),
                                                     nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels * 2,  self.upsample_path_convs[-1][-1].output_channels * 2, 1, 1, 0, 1, 1, bias=True),
                                                     nn.LeakyReLU(),
                                                     nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels * 2, num_classes, 1, 1, 0, 1, 1, bias=False))
        else:
            # convert to segmentation output
            #self.segmentation_output = nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels, num_classes, 1, 1, 0,  1, 1, False)
            self.segmentation_output = nn.Sequential(nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels, self.upsample_path_convs[-1][-1].output_channels * 2, 1, 1, 0, 1, 1, bias=True),
                                                     nn.LeakyReLU(),
                                                     nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels * 2,  self.upsample_path_convs[-1][-1].output_channels * 2, 1, 1, 0, 1, 1, bias=True),
                                                     nn.LeakyReLU(),
                                                     nn.Conv3d(self.upsample_path_convs[-1][-1].output_channels * 2, num_classes, 1, 1, 0, 1, 1, bias=False))
        # register modules
        self.downsample_path_convs = nn.ModuleList(self.downsample_path_convs)
        self.downsample_path_pooling = nn.ModuleList(self.downsample_path_pooling)
        self.upsample_path_convs = nn.ModuleList(self.upsample_path_convs)
        self.upsample_path_upsampling = nn.ModuleList(self.upsample_path_upsampling)
        self.segmentation_output = nn.ModuleList([self.segmentation_output])
        if self.uncertainty_classes > 0:
            self.upsample_path_convs_uncertainty = nn.ModuleList(self.upsample_path_convs_uncertainty)
            self.upsample_path_upsampling_uncertainy = nn.ModuleList(self.upsample_path_upsampling_uncertainy)
            self.uncertainty_output = nn.ModuleList([self.uncertainty_output])


        # run weight initialisation
        from  torch.nn.init  import kaiming_normal_, normal_
        for module in self.modules():
             if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                 kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
                 if module.bias is not None:
                    nn.init.constant_(module.bias,0)
        if self.uncertainty_classes > 0:
            for module in self.uncertainty_output[0]:
                if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,  nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
                    kaiming_normal_(module.weight, a=1e-2, nonlinearity='leaky_relu')
                   #module.weight.data.fill_(0)
                    module.bias.data.fill_(0)
            self.uncertainty_output[0][-1].weight.data.fill_(0)



    def forward(self, x):
        skip_connections = []

        for level in range(len(self.downsample_path_convs)-1):
            x = self.downsample_path_convs[level](x)
            skip_connections.append(x)
            x = self.downsample_path_pooling[level](x)

        if self.uncertainty_classes > 0:
            x1 = x
            x2 = x
            for level in range(len(self.upsample_path_upsampling)):
                # segmentation branch
                x1 = self.upsample_path_upsampling[level](x1)
                # account for differences in spatial dimension due to pooling/upsampling differences. need to look into this more
                diffx= skip_connections[- (1 + level)].shape[2] - x1.shape[2]
                diffy= skip_connections[- (1 + level)].shape[3] - x1.shape[3]
                diffz= skip_connections[- (1 + level)].shape[4] - x1.shape[4]
                x1 = F.pad(x1,[0, diffz,
                             0, diffy,
                             0, diffx])
                x1 = cat((x1,skip_connections[- (1 + level)]), dim=1)
                x1 = self.upsample_path_convs[level](x1)

                # uncertainty branch
                x2 = self.upsample_path_upsampling_uncertainy[level](x2)
                diffx= skip_connections[- (1 + level)].shape[2] - x2.shape[2]
                diffy= skip_connections[- (1 + level)].shape[3] - x2.shape[3]
                diffz= skip_connections[- (1 + level)].shape[4] - x2.shape[4]
                x2 = F.pad(x2,[0, diffz,
                             0, diffy,
                             0, diffx])
                x2 = cat((x2,skip_connections[- (1 + level)]), dim=1)
                x2 = self.upsample_path_convs_uncertainty[level](x2)
            x1 = self.segmentation_output[-1](x1)
            x2 = self.uncertainty_output[0](x2)
            return x1, x2
        else:
            for level in range(len(self.upsample_path_upsampling)):
                x = self.upsample_path_upsampling[level](x)
                # account for differences in spatial dimension due to pooling/upsampling differences. need to look into this more
                diffx= skip_connections[- (1 + level)].shape[2] - x.shape[2]
                diffy= skip_connections[- (1 + level)].shape[3] - x.shape[3]
                diffz= skip_connections[- (1 + level)].shape[4] - x.shape[4]
                x = F.pad(x,[0, diffz,
                             0, diffy,
                             0, diffx])
                x = cat((x,skip_connections[- (1 + level)]), dim=1)
                x = self.upsample_path_convs[level](x)
            x = self.segmentation_output[-1](x)
            return x
