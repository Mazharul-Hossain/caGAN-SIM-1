"""
    caGAN project data loader
"""
import os.path

import cv2 as cv
import numpy as np
import tifffile as tiff
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from utils.fcns import fix_path, prctile_norm, reorder


def data_loader(
    images_path, data_path, gt_path, 
    ny, nx, nz, batch_size, norm_flag=1,
    scale=2, wf=0, wf_path=None,
):
    if wf_path is None:
        data_loader_multi_channel_3d(
            images_path, data_path, gt_path,
            ny, nx, nz, 1, 
            batch_size, norm_flag, 
            scale, wf,
        )
    else:
        data_loader_multi_channel_3d_wf(
            images_path, data_path, wf_path, gt_path,
            ny, nx, nz, 1,
            batch_size, norm_flag,
            scale, wf,
        )


def data_loader_multi_channel_3d(
    images_path, data_path, gt_path,
    ny, nx, nz, nc,
    batch_size, norm_flag=1,
    scale=2, wf=0, wf_path=None,
):
    """

    :param images_path:
    :param data_path:
    :param gt_path:
    :param ny:
    :param nx:
    :param nz:
    :param batch_size:
    :param norm_flag:
    :param scale:
    :param wf:
    :return:
    """
    data_path = fix_path(data_path)
    gt_path = fix_path(gt_path)
    images_path = [fix_path(image_path) for image_path in images_path]
    batch_images_path = np.random.choice(images_path, size=batch_size)

    image_batch = []
    wf_batch = []
    gt_batch = []

    for path in batch_images_path:
        # print(path, data_path, gt_path, path.replace(data_path, gt_path))
        cur_img = np.zeros((nz, nx, ny, nc))
        for i in range(1, nc + 1):
            cur_img_path = os.path.join(path, f"HE_{i}.tif")
            cur_img[..., i - 1] = tiff.imread(cur_img_path)
        # cur_img = tiff.imread(path)

        cur_gt_path = path.replace(data_path, gt_path)
        cur_gt = tiff.imread(cur_gt_path + ".tif")
        cur_gt = np.expand_dims(cur_gt, axis=-1)

        n_slice = cur_img.shape[0]
        n_channels = int(n_slice / nz)
        if wf > 0:
            # cur_wf = tiff.imread(path.replace(data_path, wf_path))
            # WideField is the sum of angles and phases in each z patch
            # cur_wf = reorder(cur_img)
            cur_wf = block_reduce(
                cur_img,
                block_size=(1, 1, 1, n_channels),
                func=np.mean,
                cval=np.mean(cur_img),
            )

            # wf_shape = np.shape(cur_wf)
            # plt.figure()
            # # plot widefield images to make sure there are no fringes
            # for i in range(wf_shape[0]):
            #     plt.subplot(wf_shape[0], 1, i + 1)
            #     plt.imshow(cur_wf[i, :, :])
            # if not os.path.exists(wf_path):
            #     os.mkdir(wf_path)
            # img_name = path.split('/')[-1].split('.')[0]
            # plt.savefig(wf_path + img_name + '.png')

        if norm_flag:
            cur_img = prctile_norm(np.array(cur_img))
            cur_gt = prctile_norm(np.array(cur_gt))
            if wf > 0:
                cur_wf = prctile_norm(np.array(cur_wf))
        else:
            cur_img = np.array(cur_img) / 65535
            cur_gt = np.array(cur_gt) / 65535
            if wf > 0:
                cur_wf = np.array(cur_wf) / 65535

        image_batch.append(cur_img)
        gt_batch.append(cur_gt)
        if wf > 0:
            wf_batch.append(cur_wf)

    image_batch = np.array(image_batch)
    gt_batch = np.array(gt_batch)
    if wf > 0:
        wf_batch = np.array(wf_batch)

    # Match Input format (patch_y, patch_x, patch_z, input_channels) from (nz, nx, ny, nc)
    # nslice = image_batch.shape[1]
    # image_batch = np.reshape(image_batch, (batch_size, nslice // nz, nz, ny, nx),
    #                          order='F').transpose((0, 3, 4, 2, 1))
    image_batch = image_batch.transpose((0, 3, 2, 1, 4))
    # gt_batch = gt_batch.reshape((batch_size, nz, ny * scale, nx * scale, 1),
    #                             order='F').transpose((0, 2, 3, 1, 4))
    gt_batch = gt_batch.transpose((0, 3, 2, 1, 4))
    # wf_batch = wf_batch.reshape((batch_size, nz, ny, nx, 1),
    #                             order='F').transpose((0, 2, 3, 1, 4))
    if wf > 0:
        wf_batch = wf_batch.transpose((0, 3, 2, 1, 4))

    if wf == 1:
        image_batch = np.mean(image_batch, 4)
        for b in range(batch_size):
            image_batch[b, :, :, :] = prctile_norm(image_batch[b, :, :, :])
        image_batch = image_batch[:, :, :, np.newaxis]

    if wf > 0:
        return image_batch, wf_batch, gt_batch
    return image_batch, gt_batch


def data_loader_multi_channel_3d_wf(
    images_path, data_path, wf_path, gt_path,
    ny, nx, nz, nc,
    batch_size, norm_flag=1,
    scale=2, wf=0,
):
    return data_loader_multi_channel_3d(
        images_path, data_path, gt_path,
        ny, nx, nz, nc,
        batch_size, norm_flag,
        scale, wf, wf_path=wf_path,
    )

# ============================================================================
# Used to split training data
# ============================================================================
# import glob
# import os
# import shutil
# import numpy as np
# from sklearn.model_selection import train_test_split

# data_path = "/home/mazhar/workplace/caGAN-SIM/dataset/3D_SIM_Dataset/"
# train_path = os.path.join(data_path, "train/")
# images_path = glob.glob( train_path + "*")

# indices = np.arange(len(images_path))
# _, indices_test = train_test_split(indices, test_size=0.2, random_state=0)

# train_gt_path = os.path.join(data_path, "train_gt/")
# validate_path = os.path.join(data_path, "validate/")
# validate_gt_path = os.path.join(data_path, "validate_gt/")
# for img_path in np.array(images_path)[indices_test.astype(int)]:
#     img_name = os.path.basename(img_path)

#     shutil.move( os.path.join(train_path, img_name), os.path.join(validate_path, img_name) )
#     shutil.move( os.path.join(train_gt_path, img_name + ".tif" ), os.path.join(validate_gt_path, img_name + ".tif" ) )
