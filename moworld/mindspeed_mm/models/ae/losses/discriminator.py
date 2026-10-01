import functools
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator3D(nn.Module):
    """Defines a 3D PatchGAN discriminator as in Pix2Pix but for 3D inputs."""
    def __init__(self, input_nc=1, ndf=64, kernel_size=3, padding_size=1, n_layers=3, use_actnorm=False):
        """
        Construct a 3D PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input volumes
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            use_actnorm (bool) -- flag to use actnorm instead of batchnorm
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            raise NotImplementedError("Not implemented.")
        if isinstance(norm_layer, functools.partial):
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kernel_size, stride=2, padding=padding_size),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kernel_size, kernel_size, kernel_size), stride=(2 if n == 1 else 1, 2, 2), padding=padding_size, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=(kernel_size, kernel_size, kernel_size), stride=1, padding=padding_size, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding_size)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs):
        """Standard forward."""
        return self.main(inputs)
