import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class nnUNet(nn.Module):
    def physics_block(self, in_channels, out_channels, block_dropout=0.0):
        fcl = torch.nn.Sequential(torch.nn.Linear(in_channels, out_channels),
                                  torch.nn.SELU(),
                                  torch.nn.Linear(out_channels, out_channels),
                                  torch.nn.SELU(),
                                  torch.nn.Dropout(block_dropout).train(True),
                                  )
        return fcl

    def tiling_and_concat(self, fcl_input, conv_input, concat_axis=1):
        # Expand dimensions N times until it matches that of the conv it will be concatenated to
        expansion_scale = len(conv_input.shape) - len(fcl_input.shape)
        # expanded_input = fcl_input[(None,)*expansion_scale]
        expanded_input = fcl_input[..., None, None, None]
        # Tile across all dimensions EXCEPT dimension being concatenated to AND batch dimension: First + Second one
        tiled_fcl = expanded_input.repeat((1, 1,) + conv_input.shape[concat_axis+1:])
        # print(conv_input.shape, tiled_fcl.shape)
        physics_concat = torch.cat([conv_input, tiled_fcl], dim=concat_axis)
        # print(physics_concat.shape)
        return physics_concat

    def contracting_block(self, in_channels, out_channels, kernel_size=3, block_dropout=0.0):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
                    torch.nn.LeakyReLU(),
                    torch.nn.Dropout(block_dropout).train(True),
                )
        return block

    def expansive_block(self, in_channels, mid_channel, final_channel, kernel_size=3, block_dropout=0.0):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
                torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.BatchNorm3d(mid_channel),
                torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=final_channel, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(block_dropout).train(True),
                # torch.nn.BatchNorm3d(mid_channel),
                torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
                # torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
                # )
        return block

    def penultimate_block(self, in_channels, mid_channel, kernel_size=3, block_dropout=0.0):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
                torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
                torch.nn.LeakyReLU(),
                # torch.nn.BatchNorm3d(mid_channel),
                torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(block_dropout).train(True),
                )
        return block

    def final_block(self, mid_channel, out_channels, kernel_size=3, unc_flag=False):
        """
        This returns final block
        """
        if not unc_flag:
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.LeakyReLU(),
                    # torch.nn.BatchNorm3d(out_channels),
                    )
        else:
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
                    torch.nn.Softplus(),
                    # torch.nn.BatchNorm3d(out_channels),
            )
        return block

    def __init__(self, in_channel, out_channel, physics_flag=False, physics_input=None, physics_output=0,
                 uncertainty_flag=None, dropout_level=0.0):
        # Encode
        super(nnUNet, self).__init__()

        # Physics
        self.physics_flag = physics_flag

        # Uncertainty
        self.uncertainty_flag = uncertainty_flag
        self.dropout_level = dropout_level

        if not physics_flag:
            physics_output = 0
        if dropout_level != 0.0:
            exclusive_dropout = 0.05
        else:
            exclusive_dropout = 0.0
        # print(f'The dropout is {dropout_level}')
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=30,
                                                   block_dropout=exclusive_dropout)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(30+physics_output, 60, block_dropout=dropout_level)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(60, 120, block_dropout=dropout_level)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(120, 240, block_dropout=dropout_level)
        self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)

        if physics_flag:
            self.phys = self.physics_block(physics_input, physics_output)

        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv3d(kernel_size=3, in_channels=240, out_channels=480, padding=1),
                            torch.nn.LeakyReLU(),
                            # torch.nn.BatchNorm3d(512),
                            torch.nn.Conv3d(kernel_size=3, in_channels=480, out_channels=240, padding=1),
                            torch.nn.LeakyReLU(),
                            # torch.nn.BatchNorm3d(512),
                            # torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
                            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
                            )
        # Decode
        self.conv_decode3 = self.expansive_block(480, 240, 120, block_dropout=dropout_level)
        self.conv_decode2 = self.expansive_block(240, 120, 60, block_dropout=dropout_level)
        self.conv_decode1 = self.expansive_block(120, 60, 30, block_dropout=dropout_level)
        self.penultimate_layer = self.penultimate_block(60+physics_output, 30, block_dropout=exclusive_dropout)
        self.final_layer = self.final_block(30, out_channel, unc_flag=False)
        
        # Uncertainty
        if self.uncertainty_flag:
            self.unc_conv_decode1 = self.expansive_block(120, 60, 30, block_dropout=dropout_level)
            self.unc_penultimate_layer = self.penultimate_block(60+physics_output, 30, block_dropout=exclusive_dropout)
            self.unc_final_layer = self.final_block(30, out_channel, unc_flag=True)

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        # print(upsampled.shape, bypass.shape)
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x, physics=None):
        # Physics
        if self.physics_flag:
            physics_block = self.phys(physics)
        # Encode
        # print(f'x1 shape is {x.shape}')
        encode_block1 = self.conv_encode1(x)
        # print(f'x2 shape is {encode_block1.shape}')
        encode_pool1 = self.conv_maxpool1(encode_block1)
        
        if self.physics_flag:
            encode_pool1 = self.tiling_and_concat(physics_block, encode_pool1)
        
        # print(f'x3 shape is {encode_pool1.shape}')
        encode_block2 = self.conv_encode2(encode_pool1)
        # print(f'x4 shape is {encode_block2.shape}')
        encode_pool2 = self.conv_maxpool2(encode_block2)
        # print(f'x5 shape is {encode_pool2.shape}')
        encode_block3 = self.conv_encode3(encode_pool2)
        # print(f'x6 shape is {encode_block3.shape}')
        encode_pool3 = self.conv_maxpool3(encode_block3)
        # print(f'x7 shape is {encode_pool3.shape}')
        encode_block4 = self.conv_encode4(encode_pool3)
        # print(f'x8 shape is VIP 240 {encode_block4.shape}')
        encode_pool4 = self.conv_maxpool4(encode_block4)
        # print(f'x8a shape is {encode_pool4.shape}')
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # print(f'x9 shape is BN VIP 240 {bottleneck1.shape}')
        # Decode: Start with concat
        # print(f'bottleneck shape is {bottleneck1.shape} and the encode block shape is {encode_block4.shape}')
        decode_block4 = self.crop_and_concat(bottleneck1, encode_block4, crop=False)
        # print(f'x10 shape is 480 {decode_block4.shape}')
        cat_layer3 = self.conv_decode3(decode_block4)
        # print(f'x11 shape is 120 {cat_layer3.shape}')
        decode_block3 = self.crop_and_concat(cat_layer3, encode_block3, crop=False)
        # print(f'x12 shape is 240 {decode_block3.shape}')
        cat_layer2 = self.conv_decode2(decode_block3)
        # print(f'x13 shape is 60 {cat_layer2.shape}')
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2, crop=False)
        # print(f'x14 shape is 120 {decode_block2.shape}')
        cat_layer1 = self.conv_decode1(decode_block2)
        # print(f'x15 shape is 30 {cat_layer1.shape}')
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=False)
        # print(f'x16 shape is 60 {decode_block1.shape}')

        if self.physics_flag:
            decode_block1 = self.tiling_and_concat(physics_block, decode_block1)

        features = self.penultimate_layer(decode_block1)
        final_layer = self.final_layer(features)
        
        # Uncertainty
        if self.uncertainty_flag:
            unc_features = self.unc_penultimate_layer(decode_block1)
            unc_final_layer = self.unc_final_layer(unc_features)
            return final_layer, unc_final_layer, features
        
        # print(f'x17 shape is 2 {final_layer.shape}')
        return final_layer, features
