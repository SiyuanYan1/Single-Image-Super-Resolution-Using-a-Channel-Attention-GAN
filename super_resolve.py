import torch
from utils import *
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Model checkpoints
cagan_checkpoint = "checkpoint_srgan_final.pth.tar"
# srgan_checkpoint = "checkpoint_srgan_gefei.pth.tar"
# srgan_checkpoint="ca_.pth.tar"
srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"
srgan_checkpoint="srgan_39.pth.tar"

# Load models
srresnet = torch.load(srresnet_checkpoint,map_location=torch.device('cpu'))['model'].to(device)
srresnet.eval()
cagan_generator = torch.load(cagan_checkpoint,map_location=torch.device('cpu'))['generator'].to(device)
cagan_generator.eval()
# srgan_generator=torch.load(srgan_checkpoint,map_location=torch.device('cpu'))['generator'].to(device)
# srgan_generator.eval()



def visualize_sr(img, halve=False):
    """
    Visualizes the super-resolved images from the SRResNet and SRGAN for comparison with the bicubic-upsampled image
    and the original high-resolution (HR) image, as done in the paper.

    :param img: filepath of the HR iamge
    :param halve: halve each dimension of the HR image to make sure it's not greater than the dimensions of your screen?
                  For instance, for a 2160p HR image, the LR image will be of 540p (1080p/4) resolution. On a 1080p screen,
                  you will therefore be looking at a comparison between a 540p LR image and a 1080p SR/HR image because
                  your 1080p screen can only display the 2160p SR/HR image at a downsampled 1080p. This is only an
                  APPARENT rescaling of 2x.
                  If you want to reduce HR resolution by a different extent, modify accordingly.
    """
    # Load image, downsample to obtain low-res version
    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    if halve:
        hr_img = hr_img.resize((int(hr_img.width / 2), int(hr_img.height / 2)),
                               Image.LANCZOS)
    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)

    # Bicubic Upsampling
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)

    # Super-resolution (SR) with SRResNet
    # sr_img_srresnet = srresnet(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    # sr_img_srresnet = sr_img_srresnet.squeeze(0).cpu().detach()
    # sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')

    # Super-resolution (SR) with SRGAN
    sr_img_srgan = cagan_generator(convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device))
    sr_img_srgan = sr_img_srgan.squeeze(0).cpu().detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')

    res=cv2.imread('sr/fl.png')

    res= cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.subplot(1,4,1)
    plt.imshow(bicubic_img)
    plt.axis("off")
    plt.title("bicubic")
    plt.subplot(1, 4, 2)
    plt.imshow(res)
    plt.axis("off")
    plt.title("SRGAN")
    plt.subplot(1,4,3)
    plt.imshow(res)
    plt.axis("off")
    plt.title("Self Attention GAN")
    plt.subplot(1,4,4)
    plt.imshow(sr_img_srgan)
    plt.axis("off")
    plt.title("our Channel Attention GAN")
    plt.show()

    # s_att=cv2.imread('self/woman.png')
    # s_att= cv2.cvtColor(s_att, cv2.COLOR_BGR2RGB)
    # # plt.imshow(s_att)
    # # plt.show()
    # plt.figure()
    # sr_gan=cv2.imread('sr/woman.png')
    # sr_gan=cv2.cvtColor(sr_gan, cv2.COLOR_BGR2RGB)
    # plt.subplot(2,2,1)
    # plt.imshow(bicubic_img)
    # plt.axis("off")
    # plt.title("bicubic")
    # plt.subplot(2,2,2)
    # plt.imshow(sr_gan)
    # plt.axis("off")
    # plt.title("SRGAN")
    #
    # plt.subplot(2, 2, 3)
    # plt.imshow(s_att)
    # plt.axis("off")
    # plt.title("our Self Attention GAN")
    # plt.subplot(2,2,4)
    # plt.imshow(sr_img_srgan)
    # plt.axis("off")
    # plt.title("our Channel Attention GAN")
    #
    # plt.show()

    # # # Place SRResNet image
    #
    # grid_img.paste(sr_img_srresnet, (2 * margin + bicubic_img.width, margin))
    # text_size = font.getsize("SRResNet")
    # draw.text(
    #     xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2, margin - text_size[1] - 5],
    #     text="SRResNet", font=font, fill='black')

    # # Place SRGAN image

    # grid_img.paste(sr_img_srgan, (margin, 2 * margin))
    # text_size = font.getsize("SRGAN")
    # draw.text(
    #     xy=[margin + bicubic_img.width / 2 - text_size[0] / 2, 2 * margin + sr_img_srresnet.height - text_size[1] - 5],
    #     text="SRGAN", font=font, fill='black')

    # # # Place original HR image
    # grid_img.paste(hr_img, (2 * margin + bicubic_img.width, 2 * margin + sr_img_srresnet.height))
    # text_size = font.getsize("Original HR")
    # draw.text(xy=[2 * margin + bicubic_img.width + sr_img_srresnet.width / 2 - text_size[0] / 2,
    #               2 * margin + sr_img_srresnet.height - text_size[1] - 1], text="Original HR", font=font, fill='black')
    #
    # # Display grid
    # grid_img.show()
if __name__ == '__main__':
    # visualize_sr("test_set/Set5/bird.png")
    # visualize_sr("test_set/Set14/comic.png")#Set5/bird.png"
    visualize_sr("test_set/Set14/flowers.png")#Set5/bird.png"
    # visualize_sr('2077.png')
    # visualize_sr("test_set/BSDS100/54082.png")  # Set5/bird.png"

