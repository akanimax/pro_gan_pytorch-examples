""" Script for training of the proGAN model """

import torch as th
import data_processing.DataLoader as dl
import argparse
import yaml

from torch.backends import cudnn

# define the device for the training script
device = th.device("cuda" if th.cuda.is_available() else "cpu")

# set manual seed for the training
th.manual_seed(3)

# allow fast training using CUDNN
cudnn.benchmark = True


def parse_arguments():
    """
    command line arguments parser
    :return: args => parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="store", type=str, default="configs/1.conf",
                        help="default configuration for the Network")
    parser.add_argument("--start_depth", action="store", type=int, default=0,
                        help="Starting depth for training the network")
    parser.add_argument("--generator_file", action="store", type=str, default=None,
                        help="pretrained Generator file (compatible with my code)")
    parser.add_argument("--gen_shadow_file", action="store", type=str, default=None,
                        help="pretrained gen_shadow file")
    parser.add_argument("--discriminator_file", action="store", type=str, default=None,
                        help="pretrained Discriminator file (compatible with my code)")
    parser.add_argument("--gen_optim_file", action="store", type=str, default=None,
                        help="saved state of generator optimizer")
    parser.add_argument("--dis_optim_file", action="store", type=str, default=None,
                        help="saved_state of discriminator optimizer")

    args = parser.parse_args()

    return args


def get_config(conf_file):
    """
    parse and load the provided configuration
    :param conf_file: configuration file
    :return: conf => parsed configuration
    """
    from easydict import EasyDict as edict

    with open(conf_file, "r") as file_descriptor:
        data = yaml.load(file_descriptor)

    # convert the data into an easyDictionary
    return edict(data)


def main(args):
    """
    Main function for the script
    :param args: parsed command line arguments
    :return: None
    """

    from pro_gan_pytorch.PRO_GAN import ProGAN

    print(args.config)
    config = get_config(args.config)
    print("Current Configuration:", config)

    if config.folder_distributed:
        Dataset = dl.FoldersDistributedDataset
    else:
        Dataset = dl.FlatDirectoryImageDataset

    # create the dataset for training
    dataset = Dataset(
        data_dir=config.images_dir,
        transform=dl.get_transform(config.img_dims)
    )

    print("total examples in training: ", len(dataset))

    pro_gan = ProGAN(
        depth=config.depth,
        latent_size=config.latent_size,
        learning_rate=config.learning_rate,
        beta_1=config.beta_1,
        beta_2=config.beta_2,
        eps=config.eps,
        drift=config.drift,
        n_critic=config.n_critic,
        use_eql=config.use_eql,
        loss=config.loss_function,
        use_ema=config.use_ema,
        ema_decay=config.ema_decay,
        device=device
    )

    if args.generator_file is not None:
        print("Loading generator from:", args.generator_file)
        pro_gan.gen.load_state_dict(th.load(args.generator_file))

    if args.discriminator_file is not None:
        print("Loading discriminator from:", args.discriminator_file)
        pro_gan.dis.load_state_dict(th.load(args.discriminator_file))

    if args.gen_shadow_file is not None and config.use_ema:
        print("Loading shadow generator from:", args.gen_shadow_file)
        pro_gan.gen_shadow.load_state_dict(th.load(args.gen_shadow_file))

    if args.gen_optim_file is not None:
        print("Loading generator optimizer from:", args.gen_optim_file)
        pro_gan.gen_optim.load_state_dict(th.load(args.gen_optim_file))

    if args.dis_optim_file is not None:
        print("Loading discriminator optimizer from:", args.dis_optim_file)
        pro_gan.dis_optim.load_state_dict(th.load(args.dis_optim_file))

    print("Generator_configuration:\n%s" % str(pro_gan.gen))
    print("Discriminator_configuration:\n%s" % str(pro_gan.dis))

    # train all the networks
    pro_gan.train(
        dataset=dataset,
        epochs=config.epochs,
        fade_in_percentage=config.fade_in_percentage,
        start_depth=args.start_depth,
        batch_sizes=config.batch_sizes,
        num_workers=config.num_workers,
        feedback_factor=config.feedback_factor,
        log_dir=config.log_dir,
        sample_dir=config.sample_dir,
        num_samples=config.num_samples,
        checkpoint_factor=config.checkpoint_factor,
        save_dir=config.save_dir,
    )


if __name__ == '__main__':
    # invoke the main function of the script
    main(parse_arguments())
