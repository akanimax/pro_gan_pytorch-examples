""" Module implementing the Conditional GAN which will be trained using the Progressive growing
    technique -> https://arxiv.org/abs/1710.10196
"""

import numpy as np
import torch as th


class Generator(th.nn.Module):
    """ Generator of the GAN network """

    class InitialBlock(th.nn.Module):
        """ Module implementing the initial block of the input """

        def __init__(self, in_channels, use_eql):
            """
            constructor for the inner class
            :param in_channels: number of input channels to the block
            :param use_eql: whether to use equalized learning rate
            """
            from torch.nn import LeakyReLU
            from torch.nn.functional import local_response_norm

            super().__init__()

            if use_eql:
                from networks.CustomLayers import _equalized_conv2d, _equalized_deconv2d
                self.conv_1 = _equalized_deconv2d(in_channels, in_channels, (4, 4))
                self.conv_2 = _equalized_conv2d(in_channels, in_channels, (3, 3), pad=1)

            else:
                from torch.nn import Conv2d, ConvTranspose2d
                self.conv_1 = ConvTranspose2d(in_channels, in_channels, (4, 4))
                self.conv_2 = Conv2d(in_channels, in_channels, (3, 3), padding=1)

            # Pixelwise feature vector normalization operation
            self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2, beta=0.5,
                                                         k=1e-8)

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

    class GeneralConvBlock(th.nn.Module):
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

            super().__init__()

            self.upsample = Upsample(scale_factor=2)

            if use_eql:
                from networks.CustomLayers import _equalized_conv2d
                self.conv_1 = _equalized_conv2d(in_channels, out_channels, (3, 3), pad=1)
                self.conv_2 = _equalized_conv2d(out_channels, out_channels, (3, 3), pad=1)
            else:
                from torch.nn import Conv2d
                self.conv_1 = Conv2d(in_channels, out_channels, (3, 3), padding=1)
                self.conv_2 = Conv2d(out_channels, out_channels, (3, 3), padding=1)

            # Pixelwise feature vector normalization operation
            self.pixNorm = lambda x: local_response_norm(x, 2 * x.shape[1], alpha=2, beta=0.5,
                                                         k=1e-8)

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

    def __init__(self, depth=7, latent_size=512, use_eql=True):
        """
        constructor for the Generator class
        :param depth: required depth of the Network
        :param latent_size: size of the latent manifold
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, Upsample

        super(Generator, self).__init__()

        assert latent_size != 0 and ((latent_size & (latent_size - 1)) == 0), \
            "latent size not a power of 2"
        if depth >= 4:
            assert latent_size >= np.power(2, depth - 4), "latent size will diminish to zero"

        # state of the generator:
        self.use_eql = use_eql
        self.depth = depth
        self.latent_size = latent_size

        # register the modules required for the GAN
        self.initial_block = self.InitialBlock(self.latent_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the ToRGB layers for various outputs:
        if self.use_eql:
            from networks.CustomLayers import _equalized_conv2d
            self.toRGB = lambda in_channels: \
                _equalized_conv2d(in_channels, 3, (1, 1), bias=False)
        else:
            from torch.nn import Conv2d
            self.toRGB = lambda in_channels: Conv2d(in_channels, 3, (1, 1), bias=False)

        self.rgb_converters = ModuleList([self.toRGB(self.latent_size)])

        # create the remaining layers
        for i in range(self.depth - 1):
            if i <= 2:
                layer = self.GeneralConvBlock(self.latent_size,
                                              self.latent_size, use_eql=self.use_eql)
                rgb = self.toRGB(self.latent_size)
            else:
                layer = self.GeneralConvBlock(
                    int(self.latent_size // np.power(2, i - 3)),
                    int(self.latent_size // np.power(2, i - 2)),
                    use_eql=self.use_eql
                )
                rgb = self.toRGB(int(self.latent_size // np.power(2, i - 2)))
            self.layers.append(layer)
            self.rgb_converters.append(rgb)

        # register the temporary upsampler
        self.temporaryUpsampler = Upsample(scale_factor=2)

    def forward(self, x, depth, alpha):
        """
        forward pass of the Generator
        :param x: input noise
        :param depth: current depth from where output is required
        :param alpha: value of alpha for fade-in effect
        :return: y => output
        """
        from torch.nn.functional import tanh

        assert depth < self.depth, "Requested output depth cannot be produced"

        y = self.initial_block(x)

        if depth > 0:
            for block in self.layers[:depth - 1]:
                y = block(y)

            residual = tanh(self.rgb_converters[depth - 1](self.temporaryUpsampler(y)))
            straight = tanh(self.rgb_converters[depth](self.layers[depth - 1](y)))

            out = (alpha * straight) + ((1 - alpha) * residual)

        else:
            out = tanh(self.rgb_converters[0](y))

        return out


class Discriminator(th.nn.Module):
    """ Discriminator of the GAN """

    class FinalBlock(th.nn.Module):
        """ Initial block for the Discriminator """

        class MinibatchStdDev(th.nn.Module):
            """ module implementing the minibatch_Stddev from the Pro-GAN paper. """

            def __init__(self):
                """ constructor for the class """
                super().__init__()
                # this layer doesn't have parameters

            def forward(self, x):
                """
                forward pass of the module
                :param x: input Tensor (B x C x H x W)
                :return: fwd => output Tensor (B x (C + 1) x H x W)
                """

                # calculate the std of x over the batch dimension
                std_x = x.std(dim=0)

                # average the std over all
                m_value = std_x.mean()

                # replicate the value over all spatial locations for
                # all examples
                b_size, _, h, w = x.shape
                constant_concat = m_value.expand(b_size, 1, h, w)
                fwd = th.cat((x, constant_concat), dim=1)

                # return the output tensor
                return fwd

        def __init__(self, in_channels, use_eql):
            """
            constructor of the class
            :param in_channels: number of input channels
            :param use_eql: whether to use equalized learning rate
            """
            from torch.nn import LeakyReLU

            super().__init__()

            # declare the required modules for forward pass
            self.batch_discriminator = self.MinibatchStdDev()
            if use_eql:
                from networks.CustomLayers import _equalized_conv2d
                self.conv_1 = _equalized_conv2d(in_channels + 1, in_channels, (3, 3), pad=1)
                self.conv_2 = _equalized_conv2d(in_channels, in_channels, (4, 4))
                self.conv_3 = _equalized_conv2d(in_channels, 1, (1, 1))
            else:
                from torch.nn import Conv2d
                self.conv_1 = Conv2d(in_channels + 1, in_channels, (3, 3), padding=1)
                self.conv_2 = Conv2d(in_channels, in_channels, (4, 4))
                self.conv_3 = Conv2d(in_channels, 1, (1, 1))

            # final conv layer emulates a fully connected layer

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

    class GeneralConvBlock(th.nn.Module):
        """ General block in the discriminator  """

        def __init__(self, in_channels, out_channels, use_eql):
            """
            constructor of the class
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param use_eql: whether to use equalized learning rate
            """
            from torch.nn import AvgPool2d, LeakyReLU

            super().__init__()

            if use_eql:
                from networks.CustomLayers import _equalized_conv2d
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

    def __init__(self, height=7, feature_size=512, use_eql=True):
        """
        constructor for the class
        :param height: total height of the discriminator (Must be equal to the Generator depth)
        :param feature_size: size of the deepest features extracted
                             (Must be equal to Generator latent_size)
        :param use_eql: whether to use equalized learning rate
        """
        from torch.nn import ModuleList, AvgPool2d

        super(Discriminator, self).__init__()

        assert feature_size != 0 and ((feature_size & (feature_size - 1)) == 0), \
            "latent size not a power of 2"
        if height >= 4:
            assert feature_size >= np.power(2, height - 4), "feature size cannot be produced"

        # create state of the object
        self.use_eql = use_eql
        self.height = height
        self.feature_size = feature_size

        self.final_block = self.FinalBlock(self.feature_size, use_eql=self.use_eql)

        # create a module list of the other required general convolution blocks
        self.layers = ModuleList([])  # initialize to empty list

        # create the fromRGB layers for various inputs:
        if self.use_eql:
            from networks.CustomLayers import _equalized_conv2d
            self.fromRGB = lambda out_channels: \
                _equalized_conv2d(3, out_channels, (1, 1), bias=False)
        else:
            from torch.nn import Conv2d
            self.fromRGB = lambda out_channels: Conv2d(3, out_channels, (1, 1), bias=False)

        self.rgb_to_features = ModuleList([self.fromRGB(self.feature_size)])

        # create the remaining layers
        for i in range(self.height - 1):
            if i > 2:
                layer = self.GeneralConvBlock(
                    int(self.feature_size // np.power(2, i - 2)),
                    int(self.feature_size // np.power(2, i - 3)),
                    use_eql=self.use_eql
                )
                rgb = self.fromRGB(int(self.feature_size // np.power(2, i - 2)))
            else:
                layer = self.GeneralConvBlock(self.feature_size,
                                              self.feature_size, use_eql=self.use_eql)
                rgb = self.fromRGB(self.feature_size)

            self.layers.append(layer)
            self.rgb_to_features.append(rgb)

        # register the temporary downSampler
        self.temporaryDownsampler = AvgPool2d(2)

    def forward(self, x, height, alpha):
        """
        forward pass of the discriminator
        :param x: input to the network
        :param height: current height of operation (Progressive GAN)
        :param alpha: current value of alpha for fade-in
        :return: out => raw prediction values (WGAN-GP)
        """
        assert height < self.height, "Requested output depth cannot be produced"

        if height > 0:
            residual = self.rgb_to_features[height - 1](self.temporaryDownsampler(x))
            straight = self.layers[height - 1](
                self.rgb_to_features[height](x)
            )

            y = (alpha * straight) + ((1 - alpha) * residual)

            for block in reversed(self.layers[:height - 1]):
                y = block(y)
        else:
            y = self.rgb_to_features[0](x)

        out = self.final_block(y)

        return out


class ProGAN:
    """ Wrapper around the Generator and the Discriminator """

    def __init__(self, depth=7, latent_size=64, learning_rate=0.001, beta_1=0,
                 beta_2=0.99, eps=1e-8, drift=0.001, n_critic=1, use_eql=True,
                 loss="wgan-gp", device=th.device("cpu")):
        """
        constructor for the class
        :param depth: depth of the GAN (will be used for each generator and discriminator)
        :param latent_size: latent size of the manifold used by the GAN
        :param learning_rate: learning rate for Adam
        :param beta_1: beta_1 for Adam
        :param beta_2: beta_2 for Adam
        :param eps: epsilon for Adam
        :param n_critic: number of times to update discriminator
        :param use_eql: whether to use equalized learning rate
        :param loss: the loss function to be used
                     Can either be a string => ["wgan-gp", "wgan", "lsgan", "ralsgan"]
                     Or an instance of GANLoss
        :param device: device to run the GAN on (GPU / CPU)
        """

        from torch.optim import Adam

        # Create the Generator and the Discriminator
        self.gen = Generator(depth, latent_size, use_eql=use_eql).to(device)
        self.dis = Discriminator(depth, latent_size, use_eql=use_eql).to(device)

        # state of the object
        self.latent_size = latent_size
        self.depth = depth
        self.n_critic = n_critic
        self.use_eql = use_eql
        self.device = device
        self.drift = drift

        # define the optimizers for the discriminator and generator
        self.gen_optim = Adam(self.gen.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        self.dis_optim = Adam(self.dis.parameters(), lr=learning_rate,
                              betas=(beta_1, beta_2), eps=eps)

        # define the loss function used for training the GAN
        self.loss = self.__setup_loss(loss)

    def __setup_loss(self, loss):
        import networks.Losses as losses

        if isinstance(loss, str):
            loss = loss.lower()  # lowercase the string
            if loss == "wgan":
                loss = losses.WGAN_GP(self.device, self.dis, self.drift, use_gp=False)
            elif loss == "wgan-gp":
                loss = losses.WGAN_GP(self.device, self.dis, self.drift, use_gp=True)
            elif loss == "lsgan":
                loss = losses.LSGAN(self.device, self.dis)
            else:
                raise ValueError("Unknown loss function requested")

        elif not isinstance(loss, losses.GANLoss):
            raise ValueError("loss is neither an instance of GANLoss nor a string")

        return loss

    def optimize_discriminator(self, noise, real_batch, depth, alpha):
        """
        performs one step of weight update on discriminator using the batch of data
        :param noise: input noise of sample generation
        :param real_batch: real samples batch
        :param depth: current depth of optimization
        :param alpha: current alpha for fade-in
        :return: current loss (Wasserstein loss)
        """
        from torch.nn import AvgPool2d

        # downsample the real_batch for the given depth
        down_sample_factor = int(np.power(2, self.depth - depth - 1))
        real_samples = AvgPool2d(down_sample_factor)(real_batch)

        loss_val = 0
        for _ in range(self.n_critic):
            # generate a batch of samples
            fake_samples = self.gen(noise, depth, alpha).detach()

            loss = self.loss.dis_loss(real_samples, fake_samples, depth, alpha)

            # optimize discriminator
            self.dis_optim.zero_grad()
            loss.backward()
            self.dis_optim.step()

            loss_val += loss.item()

        return loss_val / self.n_critic

    def optimize_generator(self, noise, depth, alpha):
        """
        performs one step of weight update on generator for the given batch_size
        :param noise: input random noise required for generating samples
        :param depth: depth of the network at which optimization is done
        :param alpha: value of alpha for fade-in effect
        :return: current loss (Wasserstein estimate)
        """

        # generate fake samples:
        fake_samples = self.gen(noise, depth, alpha)

        loss = self.loss.gen_loss(fake_samples, depth, alpha)

        # optimize the generator
        self.gen_optim.zero_grad()
        loss.backward()
        self.gen_optim.step()

        # return the loss value
        return loss.item()
