from utils import *
from datasets import SRDataset
import matplotlib.pyplot as plt
import torchvision
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data
data_folder = "test_set"
test_data_names = ["Set5","Set14","BSDS100"]   #, "BSDS100,"Set14"

# Model checkpoints
srgan_checkpoint = "checkpoint_srgan_final.pth.tar"
# srgan_checkpoint = "ca_39.pth.tar"
srresnet_checkpoint = "../checkpoint_srresnet.pth.tar"

# Load model, either the SRResNet or the SRGAN
# srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
# srresnet.eval()
# model = srresnet
srgan_generator = torch.load(srgan_checkpoint,map_location=torch.device('cpu'))['generator'].to(device)
#srgan_generator = torch.load(srresnet_checkpoint,map_location=torch.device('cpu'))['model'].to(device)
srgan_generator.eval()
model = srgan_generator



# Evaluate
for test_data_name in test_data_names:
    print("\nFor %s:\n" % test_data_name)

    # Custom dataloader
    test_dataset = SRDataset(data_folder,
                             split='test',
                             crop_size=0,
                             scaling_factor=4,
                             lr_img_type='imagenet-norm',
                             hr_img_type='[-1, 1]',
                             test_data_name=test_data_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                              pin_memory=True)


    # Keep track of the PSNRs and the SSIMs across batches
    PSNRs = AverageMeter()
    SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
    with torch.no_grad():
        # Batches
        for i, (lr_imgs, hr_imgs) in enumerate(test_loader):

            # Move to default device
            lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
            hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

            # Forward prop.

            sr_img = lr_imgs.resize((hr_imgs.width, hr_imgs.height), Image.BICUBIC)

            # Calculate PSNR and SSIM
            sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
            hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel

            psnr = get_psnr(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),data_range=255)
            ssim = get_ssim(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),data_range=255.)
            PSNRs.update(psnr, lr_imgs.size(0))
            SSIMs.update(ssim, lr_imgs.size(0))
    # Print average PSNR and SSIM
    print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
    print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

print("\n")
