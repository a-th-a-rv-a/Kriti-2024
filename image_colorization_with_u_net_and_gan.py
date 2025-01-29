import os
import glob
import time
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import torch
from torch import nn, optim
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_colab = None

"""### 1.1.x Preparing Colab for running the code

If you are opening this on **Google Colab** you can uncomment and run the following to install fastai. Almost all of the code in the tutorial is with **pure PyTorch**. We need fastai here only to download part of COCO dataset and in one other step in the second section of the tutorial.

Also make sure to set your runtime to **GPU** to be able to train the models much faster.
"""

#!pip install fastai==2.4

"""The following will download about 20,000 images from COCO dataset. Notice that **we are going to use only 8000 of them** for training. Also you can use any other dataset like ImageNet as long as it contains various scenes and locations."""

from fastai.data.external import untar_data, URLs
coco_path = untar_data(URLs.COCO_SAMPLE)
coco_path = str(coco_path) + "/train_sample"
use_colab = True

if use_colab == True:
    path = coco_path
else:
    path = "Your path to the dataset"

paths = glob.glob(path + "/*.jpg") # Grabbing all the image file names
# np.random.seed(123)

paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 10000 images randomly

rand_idxs = np.random.permutation(10_000)
train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
val_idxs = rand_idxs[8000:] # choosing last 2000 as validation set

train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]
print(len(train_paths), len(val_paths))

print(len(paths),len(train_paths),len(val_paths))

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

"""Although we are using the same dataset and number of training samples, the exact 8000 images that you train your model on may vary (although we are seeding!) because the dataset here has only 20000 images with different ordering while I sampled 10000 images from the complete dataset.

### 1.2- Making Datasets and DataLoaders

I hope the code is self-explanatory. I'm resizing the images and flipping horizontally (flipping only if it is training set) and then I read an RGB image, convert it to Lab color space and separate the first (grayscale) channel and the color channels as my inputs and targets for the models  respectively. Then I'm making the data loaders.
"""

SIZE = 256
class ColorizationDataset(Dataset):
   # __init__ is the constructor of the class, which is declaring the attributes split,size,paths,transform

   #self.transforms is the new transformed image for train & for validation
   #self.split is the attribute whether the image is for training/validating
   #self.paths is the path of the image
   #self.size is the size of the image

    def __init__(self, paths, split):

        self.split = split
        self.size = SIZE
        self.paths = paths

        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((SIZE, SIZE),  Image.BICUBIC),  # Image.BICUBIC is one of the 4 way of downsampling of an image (nearest,bilinear,lanczos)
                transforms.RandomHorizontalFlip(), # A little data augmentation!
            ])
        elif split == 'val':
            self.transforms = transforms.Resize((SIZE, SIZE),  Image.BICUBIC)



    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        # •	Inside __init__, the code checks the value of split:
	      # •	If split is 'train', it assigns a transformation with resizing and random horizontal flipping to self.transforms.
	      # •	If split is 'val', it assigns a simpler transformation with only resizing to self.transforms.

        img = self.transforms(img)

        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32") # Converting RGB to L*a*b
        img_lab = transforms.ToTensor()(img_lab)

#   •	transforms.ToTensor() converts img_lab from a NumPy array to a PyTorch tensor.
# 	•	Shape Transformation:
# 	•	The shape changes from [height, width, 3] to [3, height, width].
# 	•	PyTorch tensors for image data typically follow the shape [channels, height, width] for easier processing by deep learning models.
# 	•	Value Scaling: transforms.ToTensor() also scales the pixel values from their typical LAB ranges (L: 0-100, a/b: -128 to 127) to the 0-1 range.

# Summary of Shape Changes

# 	1.	img (RGB Image as NumPy Array):
# 	•	Shape: [height, width, 3]
# 	•	Format: RGB channels as the last dimension.
# 	•	Example shape: (256, 256, 3).

# 	2.	img_lab (LAB Image as PyTorch Tensor):
# 	•	Shape: [3, height, width]
# 	•	Format: LAB channels as the first dimension.
# 	•	Example shape: (3, 256, 256).

        L = img_lab[[0], ...] / 50. - 1. # Between -1 and 1
        ab = img_lab[[1, 2], ...] / 110. # Between -1 and 1

        return {'L': L, 'ab': ab}

    def __len__(self):
        return len(self.paths)

def make_dataloaders(batch_size=16, n_workers=4, pin_memory=True, **kwargs): # A handy function to make our dataloaders
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            pin_memory=pin_memory)
    return dataloader

train_dl = make_dataloaders(paths=train_paths, split='train')
val_dl = make_dataloaders(paths=val_paths, split='val')

data = next(iter(train_dl))
Ls, abs_ = data['L'], data['ab']
print(Ls.shape, abs_.shape)
print(len(train_dl), len(val_dl))

"""### 1.3- Generator proposed by the paper

This one is a little complicated and needs explanation. This code implements a U-Net to be used as the generator of our GAN. The details of the code are out of the scope of this article but the important thing to understand is that it makes the U-Net from the middle part of it (down in the U shape) and adds down-sampling and up-sampling modules to the left and right of that middle module (respectively) at every iteration until it reaches the input module and output module. Look at the following image that I made from one of the images in the article to give you a better sense of what is happening in the code:

![unet](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/unet.png?raw=1)

The blue rectangles show the order in which the related modules are built with the code. The U-Net we will build has more layers than what is depicted in this image but it suffices to give you the idea. Also notice in the code that we are going 8 layers down, so if we start with a 256 by 256 image, in the middle of the U-Net we will get a 1 by 1 (256 / 2⁸) image and then it gets up-sampled to produce a  256 by 256 image (with two channels). This code snippet is really exciting and I highly recommend to play with it to fully grasp what every line of it is doing.
"""

class UnetBlock(nn.Module):
    def __init__(self, nf, ni, submodule=None, input_c=None, dropout=False,
                 innermost=False, outermost=False):
        super().__init__()
        self.outermost = outermost
        if input_c is None: input_c = nf
        downconv = nn.Conv2d(input_c, ni, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(ni)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(nf)

        if outermost:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(ni, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(ni * 2, nf, kernel_size=4,
                                        stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if dropout: up += [nn.Dropout(0.5)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class Unet(nn.Module):
    def __init__(self, input_c=1, output_c=2, n_down=8, num_filters=64):
        super().__init__()
        unet_block = UnetBlock(num_filters * 8, num_filters * 8, innermost=True)
        for _ in range(n_down - 5):
            unet_block = UnetBlock(num_filters * 8, num_filters * 8, submodule=unet_block, dropout=True)
        out_filters = num_filters * 8
        for _ in range(3):
            unet_block = UnetBlock(out_filters // 2, out_filters, submodule=unet_block)
            out_filters //= 2
        self.model = UnetBlock(output_c, out_filters, input_c=input_c, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)

"""### 1.4- Discriminator

The architecture of our discriminator is rather straight forward. This code implements a model by stacking blocks of Conv-BatchNorm-LeackyReLU to decide whether the input image is fake or real. Notice that the first and last blocks do not use normalization and the last block has no activation function (it is embedded in the loss function we will use).
"""

# Let’s go through the parameters num_filters=64 and n_down=3 in detail with examples. These parameters are used in the PatchDiscriminator class to control the model’s capacity (number of features learned) and the model’s depth (number of downsampling layers).

# 1. num_filters=64: Number of Filters in the First Convolutional Layer

# In convolutional neural networks (CNNs), the number of filters in a layer determines how many features the model learns at each stage. Each filter learns to detect specific patterns, textures, or details within the input data.
# 	•	num_filters=64 means the first convolutional layer will have 64 filters.
# 	•	Each filter produces a feature map, so this setting results in 64 feature maps after the first convolutional layer.
# 	•	As the model goes deeper, the number of filters typically increases, allowing the model to capture more complex features.

# Example with num_filters=64

# Let’s say we have an input image of size 256x256 with 3 channels (RGB). Here’s what happens in the first layer if we set num_filters=64:
# 	1.	First Convolutional Layer:
# 	•	Input: [3, 256, 256] (3 channels for RGB).
# 	•	Convolution with 64 filters (each of size 4x4, with stride 2 and padding 1, as specified in the code).
# 	•	Output: [64, 128, 128]
# 	•	64 feature maps: Each filter learns one feature map, resulting in 64 maps.
# 	•	128x128 spatial dimensions: Due to the stride of 2, the spatial dimensions are downsampled (reduced by half).
# 	2.	Effect of num_filters:
# 	•	Setting num_filters=64 gives the model the ability to learn 64 distinct features from the input in the first layer.
# 	•	If we increase num_filters (e.g., to 128 or 256), the model can capture more detailed and complex patterns, but it also requires more computational power and memory.

# Increasing num_filters:
# 	•	Increases model capacity: More filters mean the model can learn more complex and varied patterns.
# 	•	Increases computational cost: Each filter requires calculations, so more filters result in a higher processing time and memory usage.

# 2. n_down=3: Number of Downsampling Layers

# The number of downsampling layers (n_down) controls the depth of the discriminator network and how much spatial information is reduced as we go deeper into the network.
# 	•	n_down=3 means there will be 3 downsampling layers.
# 	•	Each downsampling layer uses a stride of 2, halving the spatial dimensions of the feature maps.
# 	•	As we go deeper and downsample more, the model’s receptive field grows, meaning each neuron can “see” a larger area of the input image, allowing it to capture larger patterns and structures.

# Example with n_down=3

# Using the same input image (256x256), let’s see how the spatial size and the number of channels change across each downsampling layer.
# 	1.	Input Image:
# 	•	Shape: [3, 256, 256] (RGB channels).

# 	2.	First Downsampling Layer:
# 	•	Input: [3, 256, 256]
# 	•	num_filters = 64 (specified for the first layer).
# 	•	Convolution with a stride of 2 and 64 filters, resulting in:
# 	•	Shape: [64, 128, 128] (64 channels and reduced spatial dimensions).

# 	3.	Second Downsampling Layer:
# 	•	Input: [64, 128, 128]
# 	•	num_filters = 128 (doubled from the previous layer).
# 	•	Convolution with a stride of 2 and 128 filters, resulting in:
# 	•	Shape: [128, 64, 64] (128 channels and reduced spatial dimensions).

# 	4.	Third Downsampling Layer:
# 	•	Input: [128, 64, 64]
# 	•	num_filters = 256 (doubled again).
# 	•	Convolution with a stride of 2 and 256 filters, resulting in:
# 	•	Shape: [256, 32, 32] (256 channels and further reduced spatial dimensions).

# 	5.	Final Layer (classification layer with 1 filter):
# 	•	Input: [256, 32, 32]
# 	•	num_filters = 1 (outputs a single channel with scores for each patch).
# 	•	Convolution with a stride of 1 and no activation/normalization, resulting in:
# 	•	Shape: [1, 32, 32] (single-channel output, each value representing a patch classification score).

# Effect of n_down

# Increasing n_down:
# 	•	Increases the model’s depth: More downsampling layers mean the model can learn more hierarchical patterns and capture larger spatial features.
# 	•	Reduces spatial dimensions: Each downsampling layer halves the spatial size, so more layers lead to a smaller feature map size.
# 	•	Enlarges the receptive field: With each layer, the model can see a larger part of the input image, capturing more global patterns.

# With n_down=3, we get three downsampling steps, reducing the 256x256 input image to a 32x32 output map. Each element in this final map represents a patch in the original image, classifying it as real or fake.

# Summary

# 	•	num_filters=64: Sets the number of filters in the first convolutional layer to 64. This determines the initial number of feature maps the model can learn, controlling the model’s capacity to capture details.
# 	•	n_down=3: Specifies 3 downsampling layers, reducing spatial dimensions and increasing the receptive field. This allows the model to learn hierarchical features and focus on larger patterns in the image.

# Together, these parameters balance the model’s ability to capture detailed features (num_filters) and learn hierarchical, larger-scale patterns (n_down).

class PatchDiscriminator(nn.Module):
    def __init__(self, input_c, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(input_c, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down-1) else 2)
                          for i in range(n_down)] # the 'if' statement is taking care of not using
                                                  # stride of 2 for the last block in this loop
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False, act=False)] # Make sure to not use normalization or
                                                                                             # activation for the last layer of the model
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True, act=True): # when needing to make some repeatitive blocks of layers,
        layers = [nn.Conv2d(ni, nf, k, s, p, bias=not norm)]          # it's always helpful to make a separate method for that purpose
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

"""Let's take a look at its blocks:"""

PatchDiscriminator(3)

"""And its output shape:"""

discriminator = PatchDiscriminator(3)
dummy_input = torch.randn(16, 3, 256, 256) # batch_size, channels, size, size
out = discriminator(dummy_input)
out.shape

"""We are using a "Patch" Discriminator here. Okay, what is it?! In a vanilla discriminator, the model outputs one number (a scaler) which represents how much the model thinks the input (which is the whole image) is real (or fake). In a patch discriminator, the model outputs one number for every patch of say 70 by 70 pixels of the input image and for each of them decides whether it is fake or not separately. Using such a model for the task of colorization seems reasonable to me because the local changes that the model needs to make are really important and maybe deciding on the whole image as in vanilla discriminator cannot take care of the subtleties of this task. Here, the model's output shape is 30 by 30 but it does not mean that our patches are 30 by 30. The actual patch size is obtained when you compute the receptive field of each of these 900 (30 multiplied by 30) output numbers which in our case will be 70 by 70.

### 1.5- GAN Loss

This is a handy class we can use to calculate the GAN loss of our final model. In the __init__ we decide which kind of loss we're going to use (which will be "vanilla" in our project) and register some constant tensors as the "real" and "fake" labels. Then when we call this module, it makes an appropriate tensor full of zeros or ones (according to what we need at the stage) and computes the loss.
"""

class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()

    def get_labels(self, preds, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real):
        labels = self.get_labels(preds, target_is_real)
        loss = self.loss(preds, labels)
        return loss

"""### 1.x Model Initialization

In the TowardsDataScince article, I didn't explain this function. Here is our logic to initialize our models. We are going to initialize the weights of our model with a mean of 0.0 and standard deviation of 0.02 which are the proposed hyperparameters in the article:
"""

def init_weights(net, init='norm', gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            if init == 'norm':
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')

            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    net.apply(init_func)
    print(f"model initialized with {init} initialization")
    return net

def init_model(model, device):
    model = model.to(device)
    model = init_weights(model)
    return model

"""### 1.6- Putting everything together

This class brings together all the previous parts and implements a few methods to take care of training our complete model. Let's investigate it.

In the __init__ we define our generator and discriminator using the previous functions and classes we defined and we also initialize them with init_model function which I didn't explain here but you can refer to my GitHub repository to see how it works. Then we define our two loss functions and the optimizers of the generator and discriminator.

The whole work is being done in optimize method of this class. First and only once per iteration (batch of training set) we call the module's forward method and store the outputs in fake_color variable of the class.

Then, we first train the discriminator by using backward_D method in which we feed the fake images produced by generator to the discriminator (make sure to detach them from the generator's graph so that they act as a constant to the discriminator, like normal images) and label them as fake. Then we feed a batch of real images from training set to the discriminator and label them as real. We add up the two losses for fake and real and take the average and then call the backward on the final loss.
Now, we can train the generator. In backward_G method we feed the discriminator the fake image and try to fool it by assigning real labels to them and calculating the adversarial loss. As I mentioned earlier, we use L1 loss as well and compute the distance between the predicted two channels and the target two channels and multiply this loss by a coefficient (which is 100 in our case) to balance the two losses and then add this loss to the adversarial loss. Then we call the backward method of the loss.
"""

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4,
                 beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_L1 = lambda_L1

        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def forward(self):
        self.fake_color = self.net_G(self.L)

    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()

        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

"""### 1.xx Utility functions

These functions were nor included in the explanations of the TDS article. These are just some utility functions to log the losses of our network and also visualize the results during training. So here you can check them out:
"""

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.count, self.avg, self.sum = [0.] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += count * val
        self.avg = self.sum / self.count

def create_loss_meters():
    loss_D_fake = AverageMeter()
    loss_D_real = AverageMeter()
    loss_D = AverageMeter()
    loss_G_GAN = AverageMeter()
    loss_G_L1 = AverageMeter()
    loss_G = AverageMeter()

    return {'loss_D_fake': loss_D_fake,
            'loss_D_real': loss_D_real,
            'loss_D': loss_D,
            'loss_G_GAN': loss_G_GAN,
            'loss_G_L1': loss_G_L1,
            'loss_G': loss_G}

def update_losses(model, loss_meter_dict, count):
    for loss_name, loss_meter in loss_meter_dict.items():
        loss = getattr(model, loss_name)
        loss_meter.update(loss.item(), count=count)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """

    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def visualize(model, data, save=True):
    model.net_G.eval()
    with torch.no_grad():
        model.setup_input(data)
        model.forward()
    model.net_G.train()
    fake_color = model.fake_color.detach()
    real_color = model.ab
    L = model.L
    fake_imgs = lab_to_rgb(L, fake_color)
    real_imgs = lab_to_rgb(L, real_color)
    fig = plt.figure(figsize=(15, 8))
    for i in range(5):
        ax = plt.subplot(3, 5, i + 1)
        ax.imshow(L[i][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 5)
        ax.imshow(fake_imgs[i])
        ax.axis("off")
        ax = plt.subplot(3, 5, i + 1 + 10)
        ax.imshow(real_imgs[i])
        ax.axis("off")
    plt.show()
    if save:
        fig.savefig(f"colorization_{time.time()}.png")

def log_results(loss_meter_dict):
    for loss_name, loss_meter in loss_meter_dict.items():
        print(f"{loss_name}: {loss_meter.avg:.5f}")

"""### 1.7- Training function

I hope this code is self-explanatory. Every epoch takes about 4 minutes on not a powerful GPU as Nvidia P5000. So if you are using 1080Ti or higher, it will be much faster.
"""

def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(val_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to
        i = 0                                  # log the losses of the complete network
        for data in tqdm(train_dl):
            model.setup_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs

model = MainModel()
train_model(model, train_dl, 100)

"""Every epoch takes about 3 to 4 minutes on Colab. After about 20 epochs you should see some reasonable results.

Okay. I let the model train for some longer (about 100 epochs). Here are the results of our baseline model:

![baseline](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/baseline.png?raw=1)

As you can see, although this baseline model has some basic understanding of some most common objects in images like sky, trees, … its output is far from something appealing and it cannot decide on the color of rare objects. It also displays some color spillovers and circle-shaped mass of color (center of first image of second row) which is not good at all. So, it seems like that with this small dataset we cannot get good results with this strategy. **Therefore, we change our strategy!**

## 2- A new strategy - the final model

Here is the focus of this article and where I'm going to explain what I did to overcome the last mentioned problem. Inspired by an idea in Super Resolution literature, I decided to pretrain the generator separately in a supervised and deterministic manner to avoid the problem of "the blind leading the blind" in the GAN game where neither generator nor discriminator knows anything about the task at the beginning of training.

Actually I use pretraining in two stages: 1- The backbone of the generator (the down sampling path) is a pretrained model for classification (on ImageNet) 2- The whole generator will be pretrained on the task of colorization with L1 loss.

In fact, I'm going to use a pretrained ResNet18 as the backbone of my U-Net and to accomplish the second stage of pretraining, we are going to train the U-Net on our training set with only L1 Loss. Then we will move to the combined adversarial and L1 loss, as we did in the previous section.

### 2.1- Using a new generator

Building a U-Net with a ResNet backbone is not something trivial so I'll use fastai library's Dynamic U-Net module to easily build one. You can simply install fastai with pip or conda (if you haven't already at the beginning of the tutorial). Here's the link to the [documentation](https://docs.fast.ai/).

#### Update Jan 8th, 2022: <br>
You need to install fastai version 2.4 for the following lines code to run w/o errors.
If you have already installed it using the cell in the beginning of the tutorial, you don't need to install it here again.
<br><br><br>
"""

# pip install fastai==2.4
from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

def build_res_unet(n_input=1, n_output=2, size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(resnet18, pretrained=True, n_in=n_input, cut=-2)
    net_G = DynamicUnet(body, n_output, (size, size)).to(device)
    return net_G

"""That's it! With just these few lines of code you can build such a complex model easily. create_body function loads the pretrained weights of the ResNet18 architecture and cuts the model to remove the last two layers (GlobalAveragePooling and a Linear layer for the ImageNet classification task). Then, DynamicUnet uses this backbone to build a U-Net with the needed output channels (2 in our case) and with an input size of 256.

### 2.2 Pretraining the generator for colorization task
"""

def pretrain_generator(net_G, train_dl, opt, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_meter.update(loss.item(), L.size(0))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.avg:.5f}")

net_G = build_res_unet(n_input=1, n_output=2, size=256)
opt = optim.Adam(net_G.parameters(), lr=1e-4)
criterion = nn.L1Loss()
pretrain_generator(net_G, train_dl, opt, criterion, 20)
#torch.save(net_G.state_dict(), "res18-unet.pt")

"""With this simple function, we pretrain the generator for 20 epochs and then we save its weights. This will take an hour on Colab. In the following section, we will use this model as the generator for our GAN and train the whole network as before:

### 2.3 Putting everything together, again!

If you want to train the model yourself, run the following cell. Instead, if you want to use the pretrained weights, skip the cell and run the one after that.
"""

net_G = build_res_unet(n_input=1, n_output=2, size=256)
net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
model = MainModel(net_G=net_G)
train_model(model, train_dl, 20)

"""Here I'm first loading the saved weights for the generator (which you have saved in the previous section) and then I'm using this model as the generator in our MainModel class  which prevents it from randomly initializing the generator. Then we train the model for 10 to 20 epochs! (compare it to the 100 epochs of the previous section when we didn't use pretraining). Each epoch takes about 3 to 4 minutes on Colab

If you are on Colab and want to use the pretrained weights, run the following cells which download the weights from my google drive and loads it to the model:
"""

# !gdown --id 1lR6DcS4m5InSbZ5y59zkH2mHt_4RQ2KV

# net_G = build_res_unet(n_input=1, n_output=2, size=256)
# net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
# model = MainModel(net_G=net_G)
# model.load_state_dict(torch.load("final_model_weights.pt", map_location=device))

"""Now, I will show the results of this final model on the test set (the black and white images that it has never seen during training) including the main title image of this article at the very beginning:

![output 1](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/main.png?raw=1)
Left: Input black & white images from test set | Right: the colorized outputs by the final model of this tutorial
---
![output2](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/img1.png?raw=1)
Left: Input black & white images from test set | Right: the colorized outputs by the final model of this tutorial
---
![output3](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/img2.png?raw=1)
Left: Input black & white images from test set | Right: the colorized outputs by the final model of this tutorial
---

## An accidental finding: You can safely remove Dropout!

Remember that when I was explaining the theory of conditional GAN in the beginning of this article, I said that the source of the noise in the architecture of the generator proposed by authors of the paper was the dropout layers. However, when I investigated the U-Net we built with the help of fastai, I did not find any dropout layers in there! Actually I first trained the final model and got the results and then I investigated the generator and found this out.

So, was the adversarial training useless? If there is no noise, how possibly the generator can have a creative effect on the output? Is it possible that the input grayscale image to the generator plays the role of noise as well? These were my exact questions at the time.

Therefor, I decided to email Dr. Phillip Isola, the first author of the same paper we implemented here, and he kindly answered these questions. According to what he said,  this conditional GAN can still work without dropout but the outputs will be more deterministic because of the lack of that noise; however, there is still enough information in that input grayscale image which enables the generator to produce compelling outputs.
Actually, I saw this in practice that the adversarial training was helpful indeed. In the next and last section, I'm going to compare the results of the pretrained U-Net with no adversarial training against the final outputs we got with adversarial training.

## Comparing the results of the pretrained U-Net with and without adversarial training

One of the cool thing I found in my experiments was that the U-Net we built with the ResNet18 backbone is already awesome in colorizing images after pretraining with L1 Loss only (a step before the final adversarial training). But, the model is still conservative and encourages using gray-ish colors when it is not sure about what the object is or what color it should be. However, it performs really awesome for common scenes in the images like sky, tree, grass, etc.

Here I show you the outputs of the U-Net without adversarial training and U-Net with adversarial training to better depict the significant difference that the adversarial training is making in our case:

![comparison](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/comparison1.png?raw=1)
(Left: pretrained U-Net without adversarial training | Right: pretrained U-Net with adversarial training)
---

You can also see the GIF below to observe the difference between the images better:

![anim](https://github.com/moein-shariatnia/Deep-Learning/blob/main/Image%20Colorization%20Tutorial/files/anim_compare.gif?raw=1)
(animation of the last two images to better see the significant difference that adversarial training is making)
---

## Final words

This project was full of important lessons for myself. I spent a lot of time during the last month to implement lots of different papers each with different strategies and it took quite a while and after A LOT of failures that I could come up with this method of training. Now you can see that how pretraining the generator significantly helped the model and improved the results.

I also learned that some observations, although at first feeling like a bad mistake of yours, are worth paying attention to and further investigation; like the case of dropout in this project. Thanks to the helpful community of deep learning and AI, you can easily ask experts and get the answer you need and become more confidant in what you were just guessing.

I want to thank the authors of this wonderful paper for their awesome work and also [the great GitHub repository of this paper](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) from which I borrowed some of the codes (with modification and simplification). I truly love the community of computer science and AI and all their hard work to improve the field and also make their contributions available to all. I'm happy to be a tiny part of this community.
"""
