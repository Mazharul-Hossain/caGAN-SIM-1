"""
    caGAN project functions
    author: Parisa Daj
"""
import numpy as np
from sewar.full_ref import uqi
from skimage.metrics import (
    mean_squared_error as compare_mse,
    normalized_root_mse as compare_nrmse,
    peak_signal_noise_ratio as compare_psnr,
    structural_similarity as compare_ssim,
)


def reorder(img, phases=5, angles=3):
    """
        Change the z data order from angles, z, phases
        to z, angles, phases
        todo: add plot_besides function
    :param img:
    :param phases:
    :param angles:
    :return:
    """
    [n_zs, n_x, n_y] = np.shape(img)
    n_z = n_zs // (angles * phases)
    five_d_img = np.reshape(img, (angles, n_z, phases, n_x, n_y))
    # swap angles with z
    new_img = five_d_img.swapaxes(1, 0)
    return np.reshape(new_img, (n_zs, n_x, n_y))


def prctile_norm(x_in, min_prc=0, max_prc=100):
    """

    :param x_in:
    :param min_prc:
    :param max_prc:
    :return: output
    """
    output = (x_in - np.percentile(x_in, min_prc)) / (
        np.percentile(x_in, max_prc) - np.percentile(x_in, min_prc) + 1e-7
    )
    output[output > 1] = 1
    output[output < 0] = 0
    return output


def img_comp(gt, pr, mses=None, nrmses=None, psnrs=None, ssims=None, uqis=None):
    """
    :param gt:
    :param pr:
    :param mses:
    :param nrmses:
    :param psnrs:
    :param ssims:
    :param uqis
    :return:
    """
    if ssims is None:
        ssims = []
    if psnrs is None:
        psnrs = []
    if nrmses is None:
        nrmses = []
    if mses is None:
        mses = []
    if uqis is None:
        uqis = []

    gt, pr = np.squeeze(gt), np.squeeze(pr)
    gt = gt.astype(np.float32)
    if gt.ndim == 2:
        num = 1
        gt = np.reshape(gt, (1, gt.shape[0], gt.shape[1]))
        pr = np.reshape(pr, (1, pr.shape[0], pr.shape[1]))
    else:
        num = np.size(gt, 0)

    for i in range(num):
        mses.append(
            compare_mse(
                prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))
            )
        )
        nrmses.append(
            compare_nrmse(
                prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))
            )
        )
        psnrs.append(
            compare_psnr(
                prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))
            )
        )
        ssims.append(
            compare_ssim(
                prctile_norm(np.squeeze(gt[i])), prctile_norm(np.squeeze(pr[i]))
            )
        )
        uqis.append(
            uqi(prctile_norm(np.squeeze(pr[i])), prctile_norm(np.squeeze(gt[i])))
        )
    return mses, nrmses, psnrs, ssims, uqis


def fix_path(path):
    """

    :param path:
    :return:
    """
    return path.replace("\\", "/")


def diffxy(img, order=3):
    """

    :param img:
    :param order:
    :return:
    """
    for _ in range(order):
        img = prctile_norm(img)
        d = np.zeros_like(img)
        dx = (img[1:-1, 0:-2] + img[1:-1, 2:]) / 2
        dy = (img[0:-2, 1:-1] + img[2:, 1:-1]) / 2
        d[1:-1, 1:-1] = img[1:-1, 1:-1] - (dx + dy) / 2
        d[d < 0] = 0
        img = d
    return img


def rm_outliers(img, order=3, thresh=0.2):
    """

    :param img:
    :param order:
    :param thresh:
    :return:
    """
    img_diff = diffxy(img, order)
    mask = img_diff > thresh
    img_rm_outliers = img
    img_mean = np.zeros_like(img)
    for i in [-1, 1]:
        for ax in range(0, 2):
            img_mean = img_mean + np.roll(img, i, axis=ax)
    img_mean = img_mean / 4
    img_rm_outliers[mask] = img_mean[mask]
    img_rm_outliers = prctile_norm(img_rm_outliers)
    return img_rm_outliers
