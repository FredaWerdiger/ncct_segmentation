# following tutorial from BRATs segmentation
import os
import pandas as pd
import sys

sys.path.append('/data/gpfs/projects/punim1086/ctp_project/MONAI/')
sys.path.append('../MONAI/')
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import matplotlib.pyplot as plt
import numpy as np
import glob
from monai.data import Dataset, CacheDataset, DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet, UNet, AttentionUnet, DenseNet
from monai.networks.layers import Norm
from torch.optim import Adam
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
    Resized,
    SaveImaged,
)

from monai_fns import *
from densenet import DenseNetFCN
from sklearn.model_selection import train_test_split

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


def main():
    HOMEDIR = os.path.expanduser('~/')
    if os.path.exists(HOMEDIR + 'mediaflux/'):
        directory = HOMEDIR + 'mediaflux/data_freda/ncct_segmentation/'
    elif os.path.exists('X:/data_freda'):
        directory = 'X:/data_freda/ncct_segmentation/'
    elif os.path.exists('/data/gpfs/projects/punim1086/ctp_project'):
        directory = '/data/gpfs/projects/punim1086/ncct_segmentation/'

    out_tag = 'model'
    if not os.path.exists(os.path.join( directory, 'out_' + out_tag)):
        os.makedirs(os.path.join( directory, 'out_' + out_tag))
    image_paths = glob.glob(os.path.join(directory, 'images', '*'))
    image_paths.sort()
    mask_paths = glob.glob(os.path.join(directory, 'masks', '*'))
    mask_paths.sort()

    ids = [os.path.basename(path).split('_mask')[0] for path in mask_paths]

    num_train = int(np.round(0.6 * len(mask_paths)))
    num_validation = int(np.round(0.2 * len(mask_paths)))
    num_test = len(ids) - (num_train + num_validation)

    random_state = 42

    train_ids, test_ids = train_test_split(ids, train_size=num_train + num_validation,
                                           test_size=num_test,
                                           random_state=random_state,
                                           shuffle=True)
    train_ids, val_ids = train_test_split(train_ids,
                                                 train_size=num_train,
                                                 test_size=num_validation,
                                                 shuffle=True)


    def make_dict(id):
        paths1 = [file for file in image_paths
                  if 'INSP_' + os.path.basename(file).split('.nii.gz')[0].split('_')[1] in id]
        paths2 = [file for file in mask_paths
                  if 'INSP_' + os.path.basename(file).split('.nii.gz')[0].split('_')[1] in id]

        files_dict = [{"image": image_name, "label": label_name} for
                      image_name, label_name in zip(paths1, paths2)]

        return files_dict

    train_files = make_dict(train_ids)
    val_files = make_dict(val_ids)
    test_files = make_dict(test_ids)

    max_epochs = 600
    image_size = [128]
    batch_size = 2
    val_interval = 2

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=image_size * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandAffined(keys=['image', 'label'], prob=0.5, rotate_range=(0, [-1, 1], [-1, 1], [-1, 1])),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=1.0),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image", "label"],
                    mode=['trilinear', "nearest"],
                    align_corners=[True, None],
                    spatial_size=image_size * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    test_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Resized(keys=["image"],
                    mode=['trilinear'],
                    align_corners=[True],
                    spatial_size=image_size * 3),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_dataset = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=1.0,
        num_workers=8)

    val_dataset = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=8)

    test_dataset = CacheDataset(
        data=test_files,
        transform=test_transforms,
        cache_rate=1.0,
        num_workers=8
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            pin_memory=True)

    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             pin_memory=True)

    s = 150
    import random
    # m = random.randint(0, len(train_files))
    # s = random.randint(100, 200)
    data_example = train_dataset[0]
    ch_in = data_example['image'].shape[0]
    # s = random.randint(5, data_example['image'].shape[2] -1)
    # plt.figure("sanity check")
    # plt.subplot(1, 2, 1)
    # plt.title(f"image")
    # plt.imshow(np.flipud(data_example["image"][0, :, :, s].detach().cpu()), cmap="gray")
    # plt.axis('off')
    # print(f"label shape: {data_example['label'].shape}")
    # plt.subplot(1, 2, 2)
    # plt.title("label")
    # plt.imshow(np.flipud(data_example["label"][0, :, :, s].detach().cpu()), cmap="gray")
    # plt.axis('off')
    # plt.show()
    # plt.close()

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

    loss_function = DiceLoss(smooth_dr=1e-5,
                             smooth_nr=0,
                             include_background=False, softmax=True, to_onehot_y=True)
    learning_rate = 1e-4
    optimizer = Adam(model.parameters(),
                     learning_rate,
                     weight_decay=1e-5)

    dice_metric = DiceMetric(include_background=False, reduction='mean')
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    epoch_loss_values = []
    dice_metric_values = []
    best_metric = -1
    best_metric_epoch = -1

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    start = time.time()
    model_path = 'best_metric_' + model._get_name() + '_' + str(max_epochs) + '.pth'
    model.train()

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
        # lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            print("Evaluating...")
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    val_outputs = model(val_inputs)

                    # compute metric for current iteration
                    # dice_metric_torch_macro(val_outputs, val_labels.long())
                    # now to for the MONAI dice metric
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    dice_metric(val_outputs, val_labels)

                mean_dice = dice_metric.aggregate().item()
                dice_metric.reset()
                dice_metric_values.append(mean_dice)

                if mean_dice > best_metric:
                    best_metric = mean_dice
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(
                        directory, 'out_' + out_tag, model_path))
                    print("saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {mean_dice:.4f}"
                    f"\nbest mean dice: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
        del loss, outputs
    end = time.time()
    time_taken = end - start
    print(f"Time taken: {round(time_taken, 0)} seconds")
    time_taken_hours = time_taken / 3600
    time_taken_mins = np.ceil((time_taken / 3600 - int(time_taken / 3600)) * 60)
    time_taken_hours = int(time_taken_hours)

    model_name = model._get_name()
    loss_name = loss_function._get_name()
    with open(
            directory + 'out_' + out_tag + '/model_info_' + str(
                max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_.txt', 'w') as myfile:
        myfile.write(f'Train dataset size: {len(train_files)}\n')
        myfile.write(f'Validation dataset size: {len(val_files)}\n')
        myfile.write(f'Test dataset size: {len(test_files)}\n')
        myfile.write(f'Model: {model_name}\n')
        myfile.write(f'Loss function: {loss_name}\n')
        myfile.write(f'Initial Learning Rate: {learning_rate}\n')
        myfile.write(f'Number of epochs: {max_epochs}\n')
        myfile.write(f'Batch size: {batch_size}\n')
        myfile.write(f'Image size: {image_size}\n')
        myfile.write(f'channels: {channels}\n')
        myfile.write(f'Validation interval: {val_interval}\n')
        myfile.write(f"Best metric: {best_metric:.4f}\n")
        myfile.write(f"Best metric epoch: {best_metric_epoch}\n")
        myfile.write(f"Time taken: {time_taken_hours} hours, {time_taken_mins} mins\n")
    # plot things
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Average Loss per Epoch")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Mean Dice (Accuracy)")
    x = [val_interval * (i + 1) for i in range(len(dice_metric_values))]
    y = dice_metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, 'b', label="Dice on validation data")
    plt.legend(loc="center right")
    plt.savefig(os.path.join(directory + 'out_' + out_tag,
                             'loss_plot_' + str(max_epochs) + '_epoch_' + model_name + '_' + loss_name + '_.png'),
                bbox_inches='tight', dpi=300, format='png')
    plt.close()

    # test

    pred_dir = os.path.join(directory + 'out_' + out_tag, "pred")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    png_dir = os.path.join(directory + 'out_' + out_tag, "pngs")
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    post_transforms = Compose([
        EnsureTyped(keys=["pred", "label"]),
        Invertd(
            keys=["pred"],
            transform=test_transforms,
            orig_keys=["image"],
            meta_keys=["pred_meta_dict"],
            orig_meta_keys=["image_meta_dict"],
            meta_key_postfix="meta_dict",
            nearest_interp=[False],
            to_tensor=[True],
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        AsDiscreted(keys="label", to_onehot=2),
        # SaveImaged(
        #     keys="pred",
        #     meta_keys="pred_meta_dict",
        #     output_dir=pred_dir,
        #     output_postfix="seg",
        #     resample=False,
        #     separate_folder=False)
    ])

    dice_metric = DiceMetric(reduction="mean", include_background=False)
    loader = LoadImage(image_only=False)
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    model.load_state_dict(torch.load(os.path.join(
        directory, 'out_' + out_tag, model_path)))

    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            test_inputs = test_data["image"].to(device)
            test_data["pred"] = model(test_inputs)

            test_data = [post_transforms(i) for i in decollate_batch(test_data)]
            test_output, test_label, test_image = from_engine(["pred", "label", "image"])(test_data)

            a = dice_metric(y_pred=test_output, y=test_label)
            original_image = loader(test_data[0]["image_meta_dict"]["filename_or_obj"])
            original_image = original_image[0]  # image data
            prediction = test_output[0][1].detach().numpy()
            subject = [name for name in ids if name in test_data[0]["image_meta_dict"]["filename_or_obj"]][0]
            save_loc = png_dir + '/' + subject + '_pred.png'

            create_image(original_image, prediction, save_loc,
                         define_zvalues(original_image), ext='png', save=True)
        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    print(f"Mean dice on test set: {metric:.4f}")
    # results['mean_dice'] = metric
    # print(results)
    # results.to_csv(directory + 'out_' + out_tag + '/results.csv', index=False)


if __name__ == '__main__':
    main()
