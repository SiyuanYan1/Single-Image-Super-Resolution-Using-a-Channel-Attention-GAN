from prettytable import PrettyTable
from models import Discriminator
from train_srgan import train
import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import Generator, Discriminator, TruncatedVGG19
from datasets import SRDataset
from utils import *
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Data parameters
data_folder = '../coco_small_small'  # train folder
print("small_small_dataset")
val_folder = 'test_set'
test_data_name = 'Set5'  # val_set
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
gpu_no = 1
# Generator parameters
EPOCH = 81

large_kernel_size_g = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels_g = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks_g = 16  # number of residual blocks
srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"  # filepath of the trained SRResNet checkpoint used for initialization

# Discriminator parameters
kernel_size_d = 3  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 4  # number of workers for loading data in the DataLoader
vgg19_i = 5  # the index i in the definition for VGG loss; see paper or models.py
vgg19_j = 4  # the index j in the definition for VGG loss; see paper or models.py
beta = 1e-3  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 500  # print training status once every __ batches
# lr = 1e-4  # learning rate

# imbalanced learning rate
lr_d = 0.0004
lr_g = 0.0001
grad_clip = None  # clip if gradients are exploding

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              n_channels=n_channels_g,
                              n_blocks=n_blocks_g,
                              scaling_factor=scaling_factor)

        # Initialize generator network with pretrained SRResNet
        srgan_checkpoint = "checkpoint_srgan.pth.tar"
        # srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"
        # generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)
        # print("---continue training at epoch 31---")
        # generator = torch.load(srgan_checkpoint, map_location=torch.device('cpu'))['generator'].to(device)
        # generator = torch.load(srresnet_checkpoint, map_location=torch.device('cpu'))['model'].to(device)
        # Initialize generator's optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr_g)

        # Discriminator
        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)

        net=discriminator
        count_parameters(net)
        train(train_loader=train_loader,
             generator=generator,
             discriminator=discriminator,
             truncated_vgg19=truncated_vgg19,
             content_loss_criterion=content_loss_criterion,
             adversarial_loss_criterion=adversarial_loss_criterion,
             optimizer_g=optimizer_g,
             optimizer_d=optimizer_d,
             epoch=epoch)
if __name__ == '__main__':
    main()
