"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from pathlib import Path
import yaml
import wandb
from qcardia_data import DataModule
from copy import deepcopy
import matplotlib.pyplot as plt


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    config_path = Path("/home/bme001/20183502/code/msc-stijn/resources/example-config_original.yaml")

    # The config contains all the model hyperparameters and training settings for the
    # experiment. Additionally, it contains data preprocessing and augmentation
    # settings, paths to data and results, and wandb experiment parameters.
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    run = wandb.init(
            project=config["experiment"]["project"],
            name=config["experiment"]["name"],
            config=config,
            save_code=True,
            mode="online",
        )

    # Get the path to the directory where the Weights & Biases run files are stored.
    online_files_path = Path(run.dir)
    print(f"online_files_path: {online_files_path}")
    datasets = []
    #Split datasets if multiple different values for key_pairs are given
    if type(wandb.config["dataset"]['subsets']) == dict:
        unique_keys = []
        unique_datasets = []
        for key, value in wandb.config["dataset"]['subsets'].items():
            if value[0] not in unique_keys:
                unique_keys.extend(value)
                unique_datasets.append([key])
            else:
                unique_datasets[unique_keys.index(value[0])].append(key)
        data_config = deepcopy(wandb.config.as_dict())
        for i in range(len(unique_datasets)):
            if '=meta' in unique_keys[i][1]:
                unique_keys[i][1] = unique_keys[i][1].split('=')[0]
                data_config['dataset']['meta_only_labels'] = True
                print(f"Meta only labels datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            elif str(unique_keys[i][1]).lower() in ['none', 'null', '']:
                data_config['dataset']['meta_only_labels'] = False
                unique_keys[i][1] = 'None'
                print(f"unlabelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            else:
                data_config['dataset']['meta_only_labels'] = False
                print(f"Labelled datasets {unique_datasets[i]} with keys {unique_keys[i]}")
            data_config['dataset']['key_pairs'] = [unique_keys[i]]
            data_config['dataset']['subsets'] = unique_datasets[i]
            data_module = DataModule(data_config)
            data_module.unique_setup()
            data_module.setup()
            datasets.append(deepcopy(data_module))
    else:
        image_key, label_key = wandb.config["dataset"]["key_pairs"][
        0
        ]  # TODO: make this more general
        data_module = DataModule(wandb.config)
        data_module.unique_setup()
        data_module.setup()
        datasets.append(data_module)
        # Get the PyTorch DataLoader objects for the training and validation datasets
    unlabelled_dataloader = None
    for data in datasets:
        if data.config['dataset']['meta_only_labels'] or data.config['dataset']['key_pairs'][0][1] == 'None':
            unlabelled_image_key = data.config['dataset']['key_pairs'][0][0]
            unlabelled_dataloader = data.train_dataloader()
            unlabelled_iter = iter(unlabelled_dataloader)
        else:
            image_key, label_key = data.config["dataset"]["key_pairs"][
                0
            ] 
            train_dataloader = data.train_dataloader()
    dataset_size = len(train_dataloader)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(train_dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            unlabelled_data = next(unlabelled_iter, None)
            if unlabelled_data is None:
                print(f"Resetting unlabelled dataloader at iteration {i}")
                unlabelled_iter = iter(unlabelled_dataloader)
                unlabelled_data = next(unlabelled_iter)
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["lge"].size(0)
            total_iters += batch_size
            epoch_iter += batch_size
            data['A'] = data['lge']
            data['B'] = unlabelled_data['lge']
            data['A_paths'] = data['meta_dict']['source']
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
                # fig, axs = plt.subplots(2, 4, figsize=(10, 5))
                # axs[0][0].imshow(model.get_current_visuals()['real_A'][0][0].cpu().detach(), cmap='gray', vmin=-2, vmax=2)
                # axs[1][0].hist(model.get_current_visuals()['real_A'][0][0].cpu().detach().numpy().flatten(), bins=100)
                # axs[0][0].set_title('Real A')
                # axs[0][0].axis('off')
                # axs[0][1].imshow(model.get_current_visuals()['real_B'][0][0].cpu().detach(), cmap='gray', vmin=-2, vmax=2)
                # axs[1][1].hist(model.get_current_visuals()['real_B'][0][0].cpu().detach().numpy().flatten(), bins=100)
                # axs[0][1].set_title('Real B')
                # axs[0][1].axis('off')
                # axs[0][2].imshow(model.get_current_visuals()['fake_B'][0][0].cpu().detach(), cmap='gray', vmin=-2, vmax=2)
                # axs[1][2].hist(model.get_current_visuals()['fake_B'][0][0].cpu().detach().numpy().flatten(), bins=100)
                # axs[0][2].set_title('Fake B')
                # axs[0][2].axis('off')
                # axs[0][3].imshow(model.get_current_visuals()['fake_A'][0][0].cpu().detach(), cmap='gray', vmin=-2, vmax=2)
                # axs[1][3].hist(model.get_current_visuals()['fake_A'][0][0].cpu().detach().numpy().flatten(), bins=100)
                # axs[0][3].set_title('Fake A')
                # axs[0][3].axis('off')
                # plt.show()

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
