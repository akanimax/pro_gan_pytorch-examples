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


class equalized_linear(th.nn.Module):
    """ Linear layer using equalized learning rate """
    def __init__(self, c_in, c_out, initializer='kaiming'):
        """
        Linear layer from pytorch extended to include equalized learning rate
        :param c_in: number of input channels
        :param c_out: number of output channels
        :param initializer: initializer to be used: one of "kaiming" or "xavier"
        """
        super(equalized_linear, self).__init__()
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
