import os
import sys
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import matplotlib.pyplot as plt
import numpy as np
import glob
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    SaveImaged,
    SplitDimd
)

import torch
import math
import os


def define_zvalues(ct_img):
    z_min = int(ct_img.shape[2] * .25)
    z_max = int(ct_img.shape[2] * .85)

    steps = int((z_max - z_min) / 18)

    if steps == 0:
        z_min = 0
        z_max = ct_img.shape[2]
        steps = 1

    z = list(range(z_min, z_max))

    rem = int(len(z) / steps) - 18

    if rem < 0:
        add_on = [z[-1] for n in range(abs(rem))]
        z.extend(add_on)
    elif rem == 0:
        z_min = z_min
        z_max = z_max
    elif rem % 2 == 0:
        z_min = z_min + int(rem / 2 * steps) + 1
        z_max = z_max - int(rem / 2 * steps) + 1

    elif rem % 2 != 0:
        z_min = z_min + math.ceil(rem / 2)
        z_max = z_max - math.ceil(rem / 2) + 1

    z = list(range(z_min, z_max, steps))

    if len(z) == 19:
        z = z[1:]
    elif len(z) == 20:
        z = z[1:]
        z = z[:18]

    return z


def create_image(ct_img,
                 pred,
                 savefile,
                 z,
                 ext='png',
                 save=False,
                 dpi=250):
    ct_img, pred = [np.rot90(im) for im in [ct_img, pred]]
    ct_img, pred = [np.fliplr(im) for im in [ct_img, pred]]
    pred = np.where(pred == 0, np.nan, pred)

    fig, axs = plt.subplots(6, 6, facecolor='k')
    fig.subplots_adjust(hspace=-0.1, wspace=-0.3)
    axs = axs.ravel()
    for ax in axs:
        ax.axis("off")
    for i in range(6):
        print(i)

        axs[i].imshow(ct_img[:, :, z[i]], cmap='gray',
                      interpolation='hanning', vmin=10, vmax=100)
        axs[i + 6].imshow(ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=100)
        im = axs[i + 6].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if 12 > len(z):
        max2 = len(z)
    else:
        max2 = 12
    for i in range(6, max2):
        print(i)
        axs[i + 6].imshow(ct_img[:, :, z[i]], cmap='gray',
                          interpolation='hanning', vmin=10, vmax=100)
        axs[i + 12].imshow(ct_img[:, :, z[i]], cmap='gray',
                           interpolation='hanning', vmin=10, vmax=100)
        im = axs[i + 12].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if not 12 > len(z):
        if len(z) > 18:
            max3 = 18
        else:
            max3 = len(z)
        for i in range(12, max3):
            print(i)
            axs[i + 12].imshow(ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=100)
            axs[i + 18].imshow(ct_img[:, :, z[i]], cmap='gray',
                               interpolation='hanning', vmin=10, vmax=100)
            axs[i + 18].imshow(pred[:, :, z[i]], cmap='Reds', interpolation='hanning', alpha=0.5, vmin=0, vmax=1)

    if savefile:
        plt.savefig(savefile, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=dpi, format=ext)
        plt.close()


def main(path_to_images, model_path):

    files = [{"image": image_name} for image_name in glob.glob(os.path.join(path_to_images, '*'))]

    image_size = [128]

    transforms = Compose(
        [
            LoadImaged(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Resized(keys="image",
                    mode='trilinear',
                    align_corners=True,
                    spatial_size=image_size * 3),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys="image"),
        ]
    )

    dataset = CacheDataset(
        data=files,
        transform=transforms,
        cache_rate=1.0,
        num_workers=8
    )

    data_example = dataset[0]
    ch_in = data_example['image'].shape[0]

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             pin_memory=True)


    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        SplitDimd(keys="pred", dim=0, keepdim=False,
                  output_postfixes=['inverted', 'pred']),
        SaveImaged(
            keys="pred",
            meta_keys="pred_meta_dict",
            output_dir=path_to_images,
            output_postfix="pred",
            resample=False,
            separate_folder=False)
    ])
    device = 'cuda'
    channels = (32, 64, 128, 256)

    model = UNet(
        spatial_dims=3,
        in_channels=ch_in,
        out_channels=2,
        channels=channels,
        strides=(2, 2, 2),
        dropout=0.2,
        num_res_units=2,
        norm=Norm.BATCH).to(device)
    # TODO: testing out channels as 1?

    model.load_state_dict(torch.load(model_path))

    model.eval()

    loader = LoadImage(image_only=False)

    with torch.no_grad():
        for i, test_data in enumerate(data_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output, test_image = from_engine(["pred", "image"])(test_data)

            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            original_image = original_image[0]  # image data
            prediction = test_output[0][1].detach().numpy()
            subject = test_data[0]["image_meta_dict"]["filename_or_obj"].split('.nii.gz')[0]
            save_loc = os.path.join(path_to_images, subject + '_pred.png')

            create_image(original_image, prediction, save_loc,
                         define_zvalues(original_image), ext='png', save=True)


if __name__ == '__main__':
    path_to_images = sys.argv[1] # location for images and for predictions
    model_path = sys.argv[2] # full path to trained model
    main(path_to_images, model_path)
