import torch as th

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from pro_gan_pytorch.PRO_GAN import ConditionalProGAN

# create the dataset:
dataset = CIFAR10("../data/cifar-10/cifar-10_with_labels",
                  transform=Compose((
                      ToTensor(),
                      Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                  )))

# create the conditional pro-gan
cond_pro_gan = ConditionalProGAN(
    num_classes=10,
    depth=4,
    device=th.device("cuda")
)

# train the model
cond_pro_gan.train(
    dataset=dataset,
    epochs=[20, 20, 20, 20],
    batch_sizes=[128, 128, 128, 128],
    fade_in_percentage=[50, 50, 50, 50],
    feedback_factor=5,
    sample_dir="training_runs/cifar-10/samples",
    save_dir="training_runs/cifar-10/models",
    log_dir="training_runs/cifar-10/models"
)
