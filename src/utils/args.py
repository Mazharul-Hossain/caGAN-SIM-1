"""
    author: Parisa Daj
    date: May 10, 2022,
    basic arguments
"""
import argparse
import random


def basic_arguments():
    """

    :return:

    todo: remove patch size from args, automate them from data dir
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="D:/Data/datasets_luhong/cropped128"
    )
    parser.add_argument("--save_weights_dir", type=str, default="../trained_models/2d/")
    parser.add_argument("--model_name", type=str, default="caGAN3D")
    parser.add_argument("--patch_y", type=int, default=128)
    parser.add_argument("--patch_x", type=int, default=128)
    parser.add_argument("--patch_z", type=int, default=1)
    parser.add_argument("--input_channels", type=int, default=15)
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument("--norm_flag", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--sample_interval", type=int, default=2)
    parser.add_argument("--validate_interval", type=int, default=5)
    parser.add_argument("--validate_num", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)

    parser.add_argument("--d_start_lr", type=float, default=1e-6)  # 2e-5
    parser.add_argument("--g_start_lr", type=float, default=1e-4)  # 1e-4

    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--load_weights", type=int, default=0)

    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--train_discriminator_times", type=int, default=1)
    parser.add_argument("--train_generator_times", type=int, default=3)

    parser.add_argument("--weight_wf_loss", type=float, default=0)
    parser.add_argument("--wave_len", type=int, default=525)
    parser.add_argument("--n_ResGroup", type=int, default=2)
    parser.add_argument("--n_RCAB", type=int, default=3)
    random.seed(10)
    return parser.parse_args()
