""" Module containing custom layers """
import torch as th


# extending Conv2D and Deconv2D layers for equalized learning rate logic
class _equalized_conv2d(th.nn.Module):
    """ conv2d with the concept of equalized learning rate """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out:  output channels
        :param k_size: kernel size (h, w) should be a tuple or a single integer
        :param stride: stride for conv
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(_equalized_conv2d, self).__init__()
        self.conv = th.nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.conv.weight, a=th.nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.conv.weight)

        self.use_bias = bias

        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.conv.weight.data ** 2)) ** 0.5
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the network
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.conv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class _equalized_deconv2d(th.nn.Module):
    """ Transpose convolution using the equalized learning rate """

    def __init__(self, c_in, c_out, k_size, stride=1, pad=0, initializer='kaiming', bias=True):
        """
        constructor for the class
        :param c_in: input channels
        :param c_out: output channels
        :param k_size: kernel size
        :param stride: stride for convolution transpose
        :param pad: padding
        :param initializer: initializer. one of kaiming or xavier
        :param bias: whether to use bias or not
        """
        super(_equalized_deconv2d, self).__init__()
        self.deconv = th.nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.deconv.weight, a=th.nn.init.calculate_gain('conv2d'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.deconv.weight)

        self.use_bias = bias

        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.deconv.weight.data ** 2)) ** 0.5
        self.deconv.weight.data.copy_(self.deconv.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale

        x = self.deconv(x.mul(dev_scale))
        if self.use_bias:
            return x + self.bias.view(1, -1, 1, 1).expand_as(x)
        return x


class _equalized_linear(th.nn.Module):
    """ Linear layer using equalized learning rate """

    def __init__(self, c_in, c_out, initializer='kaiming'):
        """
        Linear layer from pytorch extended to include equalized learning rate
        :param c_in: number of input channels
        :param c_out: number of output channels
        :param initializer: initializer to be used: one of "kaiming" or "xavier"
        """
        super(_equalized_linear, self).__init__()
        self.linear = th.nn.Linear(c_in, c_out, bias=False)
        if initializer == 'kaiming':
            th.nn.init.kaiming_normal_(self.linear.weight,
                                       a=th.nn.init.calculate_gain('linear'))
        elif initializer == 'xavier':
            th.nn.init.xavier_normal_(self.linear.weight)

        self.bias = th.nn.Parameter(th.FloatTensor(c_out).fill_(0))
        self.scale = (th.mean(self.linear.weight.data ** 2)) ** 0.5
        self.linear.weight.data.copy_(self.linear.weight.data / self.scale)

    def forward(self, x):
        """
        forward pass of the layer
        :param x: input
        :return: y => output
        """
        try:
            dev_scale = self.scale.to(x.get_device())
        except RuntimeError:
            dev_scale = self.scale
        x = self.linear(x.mul(dev_scale))
        return x + self.bias.view(1, -1).expand_as(x)


# ==========================================================
# Layers required for Building The generator and
# discriminator
# ==========================================================
class GenInitialBlock(th.nn.Module):
    """ Module implementing the initial block of the input """

    def __init__(self, in_channels, use_eql):
        """
        constructor for the inner class
        :param in_channels: number of input channels to the block
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU
        from torch.nn.functional import local_response_norm

        super(GenInitialBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3),
                                            pad=1, bias=True)

        else:
            from torch.nn import Conv2d, ConvTranspose2d
            self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4), bias=True)
            self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2 * x.shape[1],
                                                     beta=0.5, k=1e-8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input to the module
        :return: y => output
        """
        # convert the tensor shape:
        y = th.unsqueeze(th.unsqueeze(x, -1), -1)

        # perform the forward computations:
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # apply pixel norm
        y = self.pixNorm(y)

        return y


class GenGeneralConvBlock(th.nn.Module):
    """ Module implementing a general convolutional block """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor for the class
        :param in_channels: number of input channels to the block
        :param out_channels: number of output channels required
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU, Upsample
        from torch.nn.functional import local_response_norm

        super(GenGeneralConvBlock, self).__init__()

        self.upsample = Upsample(scale_factor=2)

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
            self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3),
                                            pad=1, bias=True)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, out_channels, (3, 3),
                                 padding=1, bias=True)
            self.conv_2 = Conv2d(out_channels, out_channels, (3, 3),
                                 padding=1, bias=True)

        # Pixelwise feature vector normalization operation
        self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2 * x.shape[1],
                                                     beta=0.5, k=1e-8)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the block
        :param x: input
        :return: y => output
        """
        y = self.upsample(x)
        y = self.pixNorm(self.lrelu(self.conv_1(y)))
        y = self.pixNorm(self.lrelu(self.conv_2(y)))

        return y


class MinibatchStdDev(th.nn.Module):
    """ module implementing the minibatch_Stddev from the Pro-GAN paper. """

    def __init__(self):
        """ constructor for the class """
        super(MinibatchStdDev, self).__init__()
        # this layer doesn't have parameters

    def forward(self, x):
        """
        forward pass of the module
        :param x: input Tensor (B x C x H x W)
        :return: fwd => output Tensor (B x (C + 1) x H x W)
        """

        # calculate the std of x over the batch dimension
        std_x = x.std(dim=0, unbiased=False)

        # average the std over all
        m_value = std_x.mean()

        # replicate the value over all spatial locations for
        # all examples
        b_size, _, h, w = x.shape
        constant_concat = m_value.expand(b_size, 1, h, w)
        fwd = th.cat((x, constant_concat), dim=1)

        # return the output tensor
        return fwd


class DisFinalBlock(th.nn.Module):
    """ Final block for the Discriminator """

    def __init__(self, in_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import LeakyReLU

        super(DisFinalBlock, self).__init__()

        # declare the required modules for forward pass
        self.batch_discriminator = MinibatchStdDev()
        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1)
            self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4))
            # final conv layer emulates a fully connected layer
            self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1))
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1)
            self.conv_2 = Conv2d(in_channels, in_channels, (4, 4))
            # final conv layer emulates a fully connected layer
            self.conv_3 = Conv2d(in_channels, 1, (1, 1))

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the FinalBlock
        :param x: input
        :return: y => output
        """
        # minibatch_std_dev layer
        y = self.batch_discriminator(x)

        # define the computations
        y = self.lrelu(self.conv_1(y))
        y = self.lrelu(self.conv_2(y))

        # fully connected layer
        y = self.lrelu(self.conv_3(y))  # final fully connected layer

        # flatten the output raw discriminator scores
        return y.view(-1)


class DisGeneralConvBlock(th.nn.Module):
    """ General block in the discriminator  """

    def __init__(self, in_channels, out_channels, use_eql):
        """
        constructor of the class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import AvgPool2d, LeakyReLU

        super(DisGeneralConvBlock, self).__init__()

        if use_eql:
            self.conv_1 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1)
            self.conv_2 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1)
        else:
            from torch.nn import Conv2d
            self.conv_1 = Conv2d(in_channels, in_channels, (3, 3), padding=1)
            self.conv_2 = Conv2d(in_channels, out_channels, (3, 3), padding=1)

        self.downSampler = AvgPool2d(2)

        # leaky_relu:
        self.lrelu = LeakyReLU(0.2)

    def forward(self, x):
        """
        forward pass of the module
        :param x: input
        :return: y => output
        """
        # define the computations
        y = self.lrelu(self.conv_1(x))
        y = self.lrelu(self.conv_2(y))
        y = self.downSampler(y)

        return y
