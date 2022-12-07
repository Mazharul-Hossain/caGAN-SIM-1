"""
    caGAN project: train_caGAN_3d.py
    author: Parisa Daj
    todo: think about a better way to import FairSIM PSF
"""

import datetime
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from models import caGAN3D
from utils.config import args
from utils.data_loader import (data_loader_multi_channel_3d,
                               data_loader_multi_channel_3d_wf)
from utils.fcns import img_comp
from utils.loss import create_psf_loss, loss_mse_ssim_3d
from utils.lr_controller import ReduceLROnPlateau
from utils.psf_generator import Parameters3D, cal_psf_3d, psf_estimator_3d

# Solves matplotlib memory leakage
matplotlib.use("agg")


def main():
    def write_log(l_writer, names, logs, batch_no):
        """
        :param l_writer:
        :param names:
        :param logs:
        :param batch_no:
        :return:
        """
        with l_writer.as_default():
            tf.summary.scalar(names, logs, step=batch_no)
            l_writer.flush()

    # --------------------------------------------------------------------------------
    #                             Sample and validate
    # --------------------------------------------------------------------------------
    def validate(n_iter, sample=0):
        """
        :param n_iter:
        :param sample:
        :return:
        """
        validate_path = glob.glob(validate_images_path + "*")
        validate_path.sort()
        if sample == 1:
            validate_path = np.random.choice(validate_path, size=1)
        elif validate_num < validate_path.__len__():
            validate_path = validate_path[0:validate_num]

        mses, nrmses, psnrs, ssims, uqis = [], [], [], [], []
        imgs, imgs_gt, output = [], [], []
        for path in validate_path:
            [imgs, imgs_gt] = data_loader_multi_channel_3d(
                [path],
                validate_images_path,
                validate_gt_path,
                patch_y,
                patch_x,
                patch_z,
                input_channels,
                1,
                norm_flag=norm_flag,
            )

            # output = combined.predict(imgs)
            output = g.predict(imgs)
            # predict generates [1, x, y, z, 1]
            # It is converted to [x, y, z] below
            output = np.reshape(output, (patch_x * 2, patch_y * 2, patch_z))

            output_proj = np.max(output, 2)

            gt_proj = np.max(
                np.reshape(imgs_gt, (patch_x * 2, patch_y * 2, patch_z)), 2
            )
            mses, nrmses, psnrs, ssims, uqis = img_comp(
                gt_proj, output_proj, mses, nrmses, psnrs, ssims, uqis
            )

        if sample == 0:
            # if best, save weights.best
            # combined.save_weights(save_weights_path + "weights_latest.h5")
            g.save_weights(save_weights_path + "weights_latest.h5")
            d.save_weights(save_weights_path + "weights_disc_latest.h5")

            if min(validate_nrmse) > np.mean(nrmses):
                # combined.save_weights(save_weights_path + "weights_best.h5")
                g.save_weights(save_weights_path + "weights_best.h5")
                # print("Summary:", g.summary)
                # combined.save_weights(save_weights_path + 'weights_disc_best.h5')
                d.save_weights(save_weights_path + "weights_disc_best.h5")

            validate_nrmse.append(np.mean(nrmses))
            curlr_g = lr_controller_g.on_epoch_end(n_iter, np.mean(nrmses))
            curlr_d = lr_controller_d.on_epoch_end(n_iter, np.mean(nrmses))

            write_log(writer, val_names[0], np.mean(mses), n_iter)
            write_log(writer, val_names[1], np.mean(ssims), n_iter)
            # print(n_iter, val_names[2], ":", np.mean(psnrs))
            write_log(writer, val_names[2], np.mean(psnrs), n_iter)
            # print(n_iter, val_names[3], ":", np.mean(nrmses))
            write_log(writer, val_names[3], np.mean(nrmses), n_iter)
            write_log(writer, val_names[4], np.mean(uqis), n_iter)
            write_log(writer, "lr_g", curlr_g, n_iter)
            write_log(writer, "lr_d", curlr_d, n_iter)

        else:
            imgs = np.mean(imgs, 4)
            plt.figure(figsize=(22, 6))

            # figures equal to the number of z patches in columns
            for j in range(patch_z):
                output_results = {
                    "Raw Input": imgs[0, :, :, j],
                    "Super Resolution Output": output[:, :, j],
                    "Ground Truth": imgs_gt[0, :, :, j, 0],
                }
                plt.title("Z = " + str(j))
                for img_i, (img_label, img) in enumerate(output_results.items()):
                    # first row: input image average of angles and phases
                    # second row: resulting output
                    # third row: ground truth
                    plt.subplot(3, patch_z, j + patch_z * img_i + 1)
                    plt.ylabel(img_label)
                    plt.imshow(img, cmap=plt.get_cmap("hot"))

                    plt.gca().axes.yaxis.set_ticklabels([])
                    plt.gca().axes.xaxis.set_ticklabels([])
                    plt.gca().axes.yaxis.set_ticks([])
                    plt.gca().axes.xaxis.set_ticks([])
                    plt.colorbar()

            plt.savefig(sample_path + f"{n_iter:03d}.png")  # Save sample results
            plt.close("all")  # Close figures to avoid memory leak

    # --------------------------------------------------------------------------------
    #                             Custom loss
    # --------------------------------------------------------------------------------
    def custom_binary_cross_entropy_loss(y_true, y_pred):
        """
        https://medium.com/jovianml/medical-image-reconstruction-using-generative-adversarial-networks-gans-87c2c3b224d
        Build Custom Loss Function https://cnvrg.io/keras-custom-loss-functions/
        BinaryCrossentropy https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy
        """
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        new_pred = d(y_pred)
        new_true = np.ones(new_pred.shape)
        # print(y_pred.shape, new_pred.shape, new_true.shape)

        loss = bce(new_true, new_pred)
        return loss

    # --------------------------------------------------------------------------------
    #                             Initialization
    # --------------------------------------------------------------------------------
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    data_dir = args.data_dir
    save_weights_dir = args.save_weights_dir
    validate_interval = args.validate_interval
    batch_size = args.batch_size
    d_start_lr = args.d_start_lr
    g_start_lr = args.g_start_lr
    lr_decay_factor = args.lr_decay_factor
    patch_y = args.patch_y
    patch_x = args.patch_x
    patch_z = args.patch_z
    input_channels = args.input_channels
    scale_factor = args.scale_factor
    norm_flag = args.norm_flag
    validate_num = args.validate_num
    iterations = args.iterations
    load_weights = args.load_weights
    optimizer_name = args.optimizer_name
    model_name = args.model_name
    sample_interval = args.sample_interval
    train_discriminator_times = args.train_discriminator_times
    train_generator_times = args.train_generator_times
    weight_wf_loss = args.weight_wf_loss
    wave_len = args.wave_len
    n_ResGroup = args.n_ResGroup
    n_RCAB = args.n_RCAB

    # os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = mixed_precision_training
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    # tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    data_name = data_dir.split("/")[-1]
    save_weights_name = model_name + "-SIM_" + data_name

    train_images_path = data_dir + "/training/"
    train_wf_path = data_dir + "/training_wf/"
    train_gt_path = data_dir + "/training_gt/"
    validate_images_path = data_dir + "/validate/"
    validate_gt_path = data_dir + "/validate_gt/"
    save_weights_path = save_weights_dir + "/" + save_weights_name + "/"
    sample_path = save_weights_path + "sampled_img/"

    if not os.path.exists(save_weights_path):
        os.mkdir(save_weights_path)
    if not os.path.exists(sample_path):
        os.mkdir(sample_path)

    # --------------------------------------------------------------------------------
    #                             Read OTF and PSF
    # --------------------------------------------------------------------------------
    pParam = Parameters3D()
    # 128*128*11 otf and psf numpy array
    # 525 loads FairSIM PSF
    OTF_Path = {
        488: "./OTF/3D-488-OTF-smallendian.mrc",
        560: "./OTF/3D-560-OTF-smallendian.mrc",
        525: "./OTF/splinePSF_128_128_11.mat",
    }
    psf, _ = cal_psf_3d(
        OTF_Path[wave_len],
        pParam.Ny,
        pParam.Nx,
        pParam.Nz,
        pParam.dky,
        pParam.dkx,
        pParam.dkz,
    )
    # print(np.shape(psf))

    # Find the most effective region of OTF
    sigma_y, sigma_x, sigma_z = psf_estimator_3d(psf)

    ksize = int(sigma_y * 4)
    halfy = pParam.Ny // 2
    psf = psf[halfy - ksize : halfy + ksize, halfy - ksize : halfy + ksize, :]
    psf = np.reshape(psf, (2 * ksize, 2 * ksize, pParam.Nz, 1, 1)).astype(np.float32)

    # --------------------------------------------------------------------------------
    #                           select models and optimizer
    # --------------------------------------------------------------------------------
    modelFns = {"caGAN3D": caGAN3D}
    modelFN = modelFns[model_name]
    optimizer_d = optimizers.Adam(learning_rate=d_start_lr, beta_1=0.9, beta_2=0.999)
    optimizer_g = optimizers.Adam(
        learning_rate=g_start_lr, beta_1=0.9, beta_2=0.999, clipnorm=1.0, clipvalue=10.0
    )
    optimizer_comp = optimizers.Adam(
        learning_rate=g_start_lr, beta_1=0.9, beta_2=0.999, clipnorm=1.0, clipvalue=10.0
    )

    # --------------------------------------------------------------------------------
    #                           define discriminator model
    # --------------------------------------------------------------------------------
    d = modelFN.Discriminator(
        (patch_y * scale_factor, patch_x * scale_factor, patch_z, 1)
    )
    d.compile(
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.25, from_logits=True),
        optimizer=optimizer_d,
        metrics=["binary_accuracy"],
    )

    # --------------------------------------------------------------------------------
    #                              define combined model
    # --------------------------------------------------------------------------------
    frozen_d = Model(inputs=d.inputs, outputs=d.outputs)
    frozen_d.trainable = False
    # g = modelFN.srcnn_generator((patch_y, patch_x, patch_z, input_channels))
    # g = modelFN.Generator((patch_y, patch_x, patch_z, input_channels),
    #                       n_ResGroup=n_ResGroup, n_RCAB=n_RCAB)
    g = modelFN.Generator((patch_y, patch_x, patch_z, input_channels))
    # print("g", g)

    input_lp = Input((patch_y, patch_x, patch_z, input_channels))
    fake_hp = g(input_lp)
    judge = frozen_d(fake_hp)

    if weight_wf_loss > 0:
        combined = Model(input_lp, judge)
        combined.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=optimizer_comp,
        )

        loss_wf = create_psf_loss(psf)
        g.compile(
            loss=[loss_mse_ssim_3d, loss_wf],
            optimizer=optimizer_g,
            loss_weights=[1, weight_wf_loss],
        )
    else:
        combined = Model(input_lp, [judge, fake_hp])
        combined.compile(
            loss=["binary_crossentropy", loss_mse_ssim_3d],
            optimizer=optimizer_g,
            loss_weights=[0.1, 1],
        )  # 0.1 1

    lr_controller_g = ReduceLROnPlateau(
        # model=g,
        model=combined,
        factor=lr_decay_factor,
        patience=2,
        mode="min",
        min_delta=1e-2,
        cooldown=0,
        min_learning_rate=g_start_lr * 0.01,
        verbose=1,
    )
    lr_controller_d = ReduceLROnPlateau(
        model=d,
        factor=lr_decay_factor,
        patience=2,
        mode="min",
        min_delta=1e-2,
        cooldown=0,
        min_learning_rate=d_start_lr * 0.01,
        verbose=1,
    )

    # --------------------------------------------------------------------------------
    #                                 about Tensor Board
    # --------------------------------------------------------------------------------
    log_path = save_weights_path + "graph"
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = tf.summary.create_file_writer(log_path)
    train_names = ["Generator_loss", "Discriminator_loss"]
    val_names = [
        "val_MSE", "val_SSIM",
        "val_PSNR", "val_NRMSE",
        "val_UQI",  # "test SSIM libraries",
    ]

    # --------------------------------------------------------------------------------
    #                             if exist, load weights
    # --------------------------------------------------------------------------------
    if load_weights:
        if os.path.exists(save_weights_path + "weights_best.h5"):
            g.load_weights(save_weights_path + "weights_best.h5")
            # combined.load_weights(save_weights_path + "weights_best.h5")
            d.load_weights(save_weights_path + "weights_disc_best.h5")
            print(
                "Loading weights successfully: " + save_weights_path + "weights_best.h5"
            )
        elif os.path.exists(save_weights_path + "weights_latest.h5"):
            g.load_weights(save_weights_path + "weights_latest.h5")
            # combined.load_weights(save_weights_path + "weights_latest.h5")
            d.load_weights(save_weights_path + "weights_disc_latest.h5")
            print(
                "Loading weights successfully: "
                + save_weights_path
                + "weights_latest.h5"
            )

    # --------------------------------------------------------------------------------
    #                                    training
    # --------------------------------------------------------------------------------
    # label
    batch_size_d = round(batch_size / 2)
    valid_d = np.ones(batch_size_d).reshape((batch_size_d, 1))
    fake_d = np.zeros(batch_size_d).reshape((batch_size_d, 1))
    valid = np.ones(batch_size).reshape((batch_size, 1))
    # fake = np.zeros(batch_size).reshape((batch_size, 1))

    # initialization
    start_time = datetime.datetime.now()
    g_loss_record = []
    d_loss_record = []
    lr_controller_g.on_train_begin()
    lr_controller_d.on_train_begin()
    validate_nrmse = [np.Inf]
    images_path = glob.glob(train_images_path + "*")
    images_path = [wrong_path.replace("\\", "/") for wrong_path in images_path]
    # print(images_path)

    # main training loop
    for it in range(iterations):
        local_loss_discriminator, local_loss_generator = [], []
        for _ in range(len(images_path) // 8):
            # ------------------------------------
            #         train discriminator
            # ------------------------------------
            for _ in range(train_discriminator_times):
                # input_d, gt_d = \
                #     data_loader_multi_channel_3d(images_path, train_images_path, train_gt_path,
                #                                 patch_y, patch_x, patch_z, batch_size_d,
                #                                 norm_flag=norm_flag)
                input_d, gt_d = data_loader_multi_channel_3d(
                    images_path, train_images_path, train_gt_path,
                    patch_y, patch_x, patch_z, input_channels,
                    batch_size_d, norm_flag=norm_flag,
                )
                fake_input_d = g.predict(input_d)
                # _, fake_input_d = combined.predict(input_d)
                input_d = np.concatenate((gt_d, fake_input_d), axis=0)
                label = np.concatenate((valid_d, fake_d), axis=0)
                local_loss_discriminator.append(d.train_on_batch(input_d, label))

                # discriminator loss separate for real/fake:
                # https://stackoverflow.com/questions/49988496/loss-functions-in-gans
                # loss_discriminator = d.train_on_batch(gt_d, valid_d)
                # loss_discriminator += d.train_on_batch(fake_input_d, fake_d)

            # ------------------------------------
            #         train generator
            # ------------------------------------
            for _ in range(train_generator_times):
                if weight_wf_loss > 0:
                    # input_g, wf_g, gt_g = data_loader_multi_channel_3d_wf(images_path, train_images_path,
                    #                                                     train_wf_path, train_gt_path,
                    #                                                     patch_y, patch_x, patch_z,
                    #                                                     batch_size,
                    #                                                     norm_flag=norm_flag)
                    input_g, wf_g, gt_g = data_loader_multi_channel_3d_wf(
                        images_path, train_images_path,
                        train_wf_path, train_gt_path,
                        patch_y, patch_x, patch_z, input_channels,
                        batch_size, norm_flag=norm_flag, wf=weight_wf_loss,
                    )
                    # loss_generator = g.train_on_batch(input_g, [gt_g, wf_g])
                    # print(input_g.shape, wf_g.shape, gt_g.shape)
                    # (2, 64, 64, 17, 15) (2, 64, 64, 17, 15) (2, 128, 128, 17, 1)

                    b = g.train_on_batch(input_g, [gt_g, wf_g])
                    a = combined.train_on_batch(input_g, valid)
                    local_loss_generator.append([a, b])
                    # print("local_loss_generator:", a, b, local_loss_generator)

                else:
                    # input_g, gt_g = data_loader_multi_channel_3d(images_path,
                    #                                             train_images_path, train_gt_path,
                    #                                             patch_y, patch_x, patch_z, batch_size,
                    #                                             norm_flag=norm_flag)
                    input_g, gt_g = data_loader_multi_channel_3d(
                        images_path, train_images_path, train_gt_path,
                        patch_y, patch_x, patch_z, input_channels,
                        batch_size, norm_flag=norm_flag,
                    )
                    # loss_generator = g.train_on_batch(input_g, gt_g)
                    local_loss_generator.append(combined.train_on_batch(input_g, gt_g))

        loss_discriminator = np.mean(local_loss_discriminator, axis=0)
        d_loss_record.append(loss_discriminator[0])

        loss_generator = np.mean(local_loss_generator, axis=0)
        g_loss_record.append(loss_generator)

        elapsed_time = datetime.datetime.now() - start_time

        # print("loss_discriminator:", ", ".join(map(str, loss_discriminator)))
        # print("loss_generator:", ", ".join(map(str, loss_generator)))
        print(
            f"epoch {(it + 1):03d} time: {str(elapsed_time).split('.')[0]}, "
            f"d_loss = {loss_discriminator[0]:.5f}, "
            f"d_acc = {loss_discriminator[1]:.5f}, "
            f"g_loss = {loss_generator[0]:.3f} {loss_generator[1]:.3f}"
        )

        if (it + 1) % sample_interval == 0:
            # images_path = glob.glob(train_images_path + "*")
            validate(it + 1, sample=1)

        if (it + 1) % validate_interval == 0:
            validate(it + 1, sample=0)
            write_log(writer, train_names[0], np.mean(g_loss_record), it + 1)
            write_log(writer, train_names[1], np.mean(d_loss_record), it + 1)
            g_loss_record = []
            d_loss_record = []


if __name__ == "__main__":
    main()
