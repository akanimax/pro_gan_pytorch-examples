# pro_gan_pytorch-examples
This repository contains examples trained using 
the python package `pro-gan-pth`. You can find the github repo for
the project at
[github-repository](https://github.com/akanimax/pro_gan_pytorch)
and the PyPI package at 
[pypi](https://pypi.org/project/pro-gan-pth/) <br><br>

There are two examples presented here for <b>LFW</b> dataset and
<b>MNIST</b> dataset. Please refer to the following sections 
for how to train and / or load the provided trained weights 
for these models.

## Prior Setup
Before running any of the following training experiments, please setup
your `VirtualEnv` with the required packages for this project. Importantly,
please install the progan package using `$ pip install pro-gan-pth` and 
your appropriate <b> gpu / cpu </b> version of `PyTorch 0.4.0`. Once this 
is done, you can proceed with the following experiments.

## LFW Experiment
The configuration used for the LFW training experiment can be found in
`implementation/configs/lfw.conf` in this repository.  The training was
performed using the `wgan-gp` loss function. <br>

<h4> Examples: </h4>
<p align="center">
<img align="center" src ="https://raw.githubusercontent.com/akanimax/pro_gan_pytorch-examples/master/implementation/training_runs/lfw/generated_samples/medium_2.gif" height=60% width=60%/>
</p>
<br>

<h4> Sample loss plot: </h4>
<img align="center" src ="https://raw.githubusercontent.com/akanimax/pro_gan_pytorch-examples/master/implementation/training_runs/lfw/losses/loss_plots/loss_for_4_x_4.png"/>
<br>


## MNIST Experiment
The configuration used for the MNIST training experiment can be found in
`implementation/configs/mnist.conf` in this repository. The training was
performed using the `lsgan` loss function. <br>
<h4> Examples: </h4>
<p align="center">
<img align="center" src ="https://raw.githubusercontent.com/akanimax/pro_gan_pytorch-examples/master/implementation/training_runs/mnist/generated_samples/gen_3_12_651.png"/>
</p>
<br>

<h4> Sample loss plot: </h4>
<img src ="https://raw.githubusercontent.com/akanimax/pro_gan_pytorch-examples/master/implementation/training_runs/mnist/losses/loss_plots/loss_for_4_x_4.png"/>
<br>

## How to use:
<h4> Running the training script: </h4>
For running the training script, simply use the following procedure:

    $ cd implementation
    $ python train_network.py --config=configs/mnist.conf
    
You can tinker with the configuration for your desired behaviour. 
This training script also exposes some of the use cases of the package's
api.

<h4> Generating loss plots: </h4>
You can generate the loss plots from the `loss-logs` by using the provided
script. The logs get generated while the training is in progress.

    $ python generate_loss_plots --logdir=training_runs/mnist/losses/ \
                                 --plotdir=training_runs/mnist/losses/loss_plots/


<h4> Using trained model: </h4>
please refer to the following code snippet if you just wish to use
the trained model for generating samples:
    
    import torch as th
    import pro_gan_pytorch.PRO_GAN as pg
    import matplotlib.pyplot as plt

    device = th.device("cuda" if th.cuda.is_available() 
                       else "cpu")
    gen = pg.Generator(depth=4, latent_size=128, 
                       use_eql=False).to(device)

    gen.load_state_dict(
        th.load("training_runs/saved_models/GAN_GEN_3.pth")
    )

    noise = th.randn(1, 128).to(device)
    
    sample_image = gen(noise, detph=3, alpha=1).detach()
    
    plt.imshow(sample_image[0].permute(1, 2, 0) / 2 + 0.5)
    plt.show()
    
The trained weights can be found in the `saved_models` 
directory present in respective `training_runs`.

## Thanks:
Please feel free to open PRs here if you train on other datasets 
using this package. <br>

Best regards, <br>
@akanimax :)sion
