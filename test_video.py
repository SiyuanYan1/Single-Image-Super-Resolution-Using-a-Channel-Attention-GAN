import argparse
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor, ToPILImage
# from tqdm import tqdm
from utils import *
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test Single Video')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--video_name', default='test.mp4',type=str, help='test low resolution video name')
    parser.add_argument('--model_name', default='netG_epoch_4_100.pth', type=str, help='generator model epoch name')
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    UPSCALE_FACTOR = opt.upscale_factor
    VIDEO_NAME = opt.video_name
    MODEL_NAME = opt.model_name
    srgan_checkpoint = "checkpoint_srgan_final.pth.tar"
    model = torch.load(srgan_checkpoint,map_location=torch.device(device))['generator'].to(device)
    model.eval()
    # model = Generator(UPSCALE_FACTOR).eval()
    # if torch.cuda.is_available():
    #     model = model.cuda()
    # # for cpu
    # # model.load_state_dict(torch.load('epochs/' + MODEL_NAME, map_location=lambda storage, loc: storage))
    # model.load_state_dict(torch.load('epochs/' + MODEL_NAME))

    videoCapture = cv2.VideoCapture(VIDEO_NAME)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frame_numbers = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    sr_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) ),
                     int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)) )
    compared_video_size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)  * 2 + 10),
                           int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))  + 10 + int(
                               int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) * 2 + 10) / int(
                                   10 * int(int(
                                       videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH) ) // 5 + 1)) * int(
                                   int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)) // 5 - 9)))
    # print("sr_size vs compare:",sr_video_size,compared_video_size)
    output_sr_name = 'out_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    output_compared_name = 'compare_srf_' + str(UPSCALE_FACTOR) + '_' + VIDEO_NAME.split('.')[0] + '.avi'
    sr_video_writer = cv2.VideoWriter(output_sr_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps, sr_video_size)
    compared_video_writer = cv2.VideoWriter(output_compared_name, cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), fps,
                                            compared_video_size)
    # read frame
    success, frame = videoCapture.read()
    # test_bar = tqdm(range(int(frame_numbers)), desc='[processing video and saving result videos]')
    for index in range(int(frame_numbers)):
        print(index,'/',frame_numbers)
        if success:
            # image = Variable(ToTensor()(frame), volatile=True).unsqueeze(0)
            # if torch.cuda.is_available():
            #     image = image.cuda()
            # print(frame.shape)
            image=np.array(frame, dtype=np.uint8)
            image=Image.fromarray(image)
            image=image.convert('RGB')
            bicubic = image.resize((int(image.width / 4), int(image.height / 4)),
                                   Image.BICUBIC)
            # image=Image.fromarray(frame)  #640x272

            # print("before",bicubic)
            # image.save('a.png')
            # plt.figure()
            # plt.subplot(1, 2, 1)

            # plt.imshow(bicubic)
            image = convert_image(bicubic, source='pil', target='imagenet-norm').unsqueeze(0)
            out = model(image.to(device))
            out = out.squeeze(0).cpu().detach()

            out_img = convert_image(out, source='[-1, 1]', target='pil')#2560x1088
            # plt.subplot(1, 2, 2)
            # print('after',out_img)
            # plt.imshow(out_img)
            # plt.show()
            out_img=np.array(out_img)
            sr_video_writer.write(out_img)
            # make compared video and crop shot of left top\right top\center\left bottom\right bottom
            out_img = ToPILImage()(out_img)
            crop_out_imgs = transforms.FiveCrop(size=out_img.width // 5 - 9)(out_img)
            crop_out_imgs = [np.asarray(transforms.Pad(padding=(10, 5, 0, 0))(img)) for img in crop_out_imgs]
            out_img = transforms.Pad(padding=(5, 0, 0, 5))(out_img)

            # compared_img = transforms.Resize(size=(sr_video_size[1], sr_video_size[0]), interpolation=Image.BICUBIC)(
            #     ToPILImage()(frame))
            # print('c',compared_img)

            compared_img=bicubic.resize((sr_video_size[0],sr_video_size[1] ))
            # print(compared_img)
            # print(out_img)
            # plt.imshow(compared_img)
            # plt.show()
            crop_compared_imgs = transforms.FiveCrop(size=compared_img.width // 5 - 9)(compared_img)
            crop_compared_imgs = [np.asarray(transforms.Pad(padding=(0, 5, 10, 0))(img)) for img in crop_compared_imgs]
            compared_img = transforms.Pad(padding=(0, 0, 5, 5))(compared_img)
            # concatenate all the pictures to one single picture
            print('vs',compared_img,out_img)
            top_image = np.concatenate((np.asarray(compared_img), np.asarray(out_img)), axis=1)
            bottom_image = np.concatenate(crop_compared_imgs + crop_out_imgs, axis=1)
            bottom_image = np.asarray(transforms.Resize(
                size=(int(top_image.shape[1] / bottom_image.shape[1] * bottom_image.shape[0]), top_image.shape[1]))(
                ToPILImage()(bottom_image)))
            final_image = np.concatenate((top_image, bottom_image))
            # save compared video
            compared_video_writer.write(final_image)
            # next frame
            success, frame = videoCapture.read()