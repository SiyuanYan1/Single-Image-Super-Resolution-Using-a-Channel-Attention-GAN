import argparse
import time
from utils import *
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt
# from model import Generator

parser = argparse.ArgumentParser(description='Test Single Image')
parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
parser.add_argument('--test_mode', default='CPU', type=str, choices=['GPU', 'CPU'], help='using GPU or CPU')
parser.add_argument('--image_name', type=str, help='test low resolution image name')
parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
opt = parser.parse_args()

UPSCALE_FACTOR = opt.upscale_factor
TEST_MODE = True if opt.test_mode == 'GPU' else False
# IMAGE_NAME = "test_set/Set5/baby.png"
# IMAGE_NAME="test_set/Set14/flowers.png"
# IMAGE_NAME="2077.png"
IMAGE_NAME='cyberpunk4.png'
srgan_checkpoint = "checkpoint_srgan_final.pth.tar"
srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"
srresnet = torch.load(srresnet_checkpoint,map_location=torch.device('cpu'))['model']
srresnet.eval()
srgan_generator = torch.load(srgan_checkpoint,map_location=torch.device('cpu'))['generator']
srgan_generator.eval()

image = Image.open(IMAGE_NAME,mode='r')

image=image.convert('RGB')
print(image)
# image = Variableimage), volatile=True).unsqueeze(0)
bicubic = image.resize((int(image.width / 4), int(image.height / 4)),
                           Image.BICUBIC)
print('bo',bicubic)
#.resize((int(image.width ), int(image.height)),Image.BICUBIC)
lr_img=convert_image(bicubic, source='pil', target='imagenet-norm').unsqueeze(0)
print(lr_img.size())

# if TEST_MODE:
#     image = image.cuda()

start = time.clock()
out = srgan_generator(lr_img)
out = out.squeeze(0).cpu().detach()
out = convert_image(out, source='[-1, 1]', target='pil')
print('out',out)

plt.figure()

plt.subplot(1, 2, 1)
plt.imshow(bicubic)
plt.axis("off")
plt.title("4x downsampled lr image")
plt.subplot(1, 2, 2)
plt.imshow(out)
plt.axis("off")
plt.title("channel attention gan")
# plt.subplot(1, 3, 3)
# plt.imshow(image)
# plt.axis("off")
# plt.title("original")
# plt.show()
# plt.imshow(out)
plt.show()
elapsed = (time.clock() - start)
print('cost' + str(elapsed) + 's')
# out_img = ToPILImage()(out[0].data.cpu())
# out_img.save('test_set/im_{}_{}.png'.format(str(1), str(1)))
