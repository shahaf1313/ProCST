import argparse
import datetime
import random
import numpy as np
import os
import sys

def get_arguments():
    parser = argparse.ArgumentParser()

    # load, input, save configurations:
    parser.add_argument("--gpus", type=int, nargs='+', help="String that contains available GPUs to use", default=[0])
    parser.add_argument('--manual_seed', default=1337, type=int, help='manual seed')
    parser.add_argument('--continue_train_from_path', type=str, help='Path to folder that contains all networks and continues to train from there', default='')
    parser.add_argument('--resume_to_epoch', default=1, type=int, help='Resumes training from specified epoch')
    parser.add_argument('--resume_step', default=1, type=int, help='Resumes Semseg training to specified step')
    parser.add_argument('--nc_im', type=int, help='image # channels', default=3)
    parser.add_argument('--no_drop_last', default=True, action='store_false', help='When this flag turns on, last batch is not dropped. Regular behavour: drops last batch in training mode.')

    #SiT dataset configurations:
    parser.add_argument("--sit_output_path", type=str, default=None, help="Path to output SiT generated dataset")
    parser.add_argument("--trained_procst_path", type=str, default=None, help="Path to pretrained ProCST model.")
    parser.add_argument('--skip_created_files', default=False, action='store_true', help='Skip already created files in the output directory.')

    # Dataset parameters:
    parser.add_argument("--source", type=str, default='gta5', help='Source dataset. gta5/synthia/zerowastev1')
    parser.add_argument("--target", type=str, default='cityscapes', help='target dataset. cityscapes/zerowastev2')
    parser.add_argument("--src_data_dir", type=str, default='data/gta', help='Path to the directory containing the source dataset.')
    parser.add_argument("--trg_data_dir", type=str, default='data/cityscapes', help='Path to the directory containing the target dataset.')
    parser.add_argument("--num_workers", type=int, default=6, help="Number of threads for each worker")

    # generator parameters:
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--batch_size_list', type=int, nargs='+', help="batch size in each one of the scales", default=[0])
    parser.add_argument('--warmup_epochs', type=int, default=0, help='Number of warmup epochs before switching to label conditioned generator.')
    parser.add_argument('--ker_size', type=int, help='kernel size', default=3)
    parser.add_argument('--body_layers', type=int, help='Number of layers in the body of the generator/discriminator. For low scales, this number is decreased by 2.', default=5)
    parser.add_argument('--stride', help='stride', default=1)
    parser.add_argument('--padd_size', type=int, help='net pad size', default=1)
    parser.add_argument('--groups_num', type=int, help='number of groups in Group Norm', default=8)
    parser.add_argument('--base_channels_list', type=int, nargs='+', help='number of channels in the generator and discriminator.', default=[64,64,64])

    # pyramid parameters:
    parser.add_argument('--scale_factor', type=float, help='pyramid scale factor', default=0.5)
    parser.add_argument('--num_scales', type=int, help='number of scales in the pyramid', default=2)

    # optimization parameters:
    parser.add_argument('--epochs_per_scale', type=int, default=40, help='number of epochs to train per scale')
    parser.add_argument('--gamma', type=float, help='scheduler gamma', default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--lr_d', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')

    # loss parameters:
    parser.add_argument('--lambda_grad', type=float, help='gradient penelty weight', default=0.1)
    parser.add_argument('--lambda_adversarial', type=float, help='adversarial loss weight', default=1)
    parser.add_argument('--lambda_cyclic', type=float, help='cyclic loss weight', default=1)
    parser.add_argument('--lambda_labels', type=float, help='label loss weight', default=1)
    parser.add_argument('--lambda_style', type=float, help='Style loss weight', default=10)
    parser.add_argument('--content_layers', type=int, nargs='+', help='Layer indices to extract content features', default=[15])
    parser.add_argument('--style_layers', type=int, nargs='+', help='Layer indices to extract style features', default=[3, 8, 15, 22])
    parser.add_argument('--content_weight', type=float, help='Content loss weight', default=1.0/30)
    parser.add_argument('--style_weight', type=float, help='style loss weight', default=30.0/30)
    parser.add_argument('--tv_weight', type=float, help='tv loss weight', default=1.0/30)

    # Semseg network parameters:
    parser.add_argument("--no_semseg", default=False, action='store_true', help="Disables semseg loss (cyclic label loss, CE on SIT images) at last scale.")
    parser.add_argument("--model", type=str, required=False, default='DeepLabV2', help="available options : DeepLabV2, DeepLab and VGG")
    parser.add_argument("--num_classes", type=int, required=False, default=19, help="Number of classes in the segmentation task. Default - 19")
    parser.add_argument('--lr_semseg', type=float, default=0.0001, help='learning rate, default=0.0001')
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Regularisation parameter for L2-loss.")

    # deeplabV1 parameters:
    parser.add_argument("--ita", type=float, default=2.0, help="ita for robust entropy")
    parser.add_argument("--entW", type=float, default=0.005, help="weight for entropy")
    parser.add_argument("--power", type=float, default=0.9, help="Decay parameter to compute the learning rate (only for deeplab).")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum component of the optimiser.")

    # Miscellaneous parameters:
    parser.add_argument("--tb_logs_dir", type=str, required=False, default='./runs', help="Path to Tensorboard logs dir.")
    parser.add_argument('--debug_run', default=False, action='store_true')
    parser.add_argument('--debug_stop_iteration', type=int, default=15, help='Iteration number to finish training current scale in debug mode.')
    parser.add_argument('--debug_stop_epoch', type=int, default=0, help='Epoch number to finish training current scale in debug mode.')
    parser.add_argument("--checkpoints_dir", type=str, required=False, default='./TrainedModels', help="Where to save snapshots of the model.")
    parser.add_argument("--print_rate", type=int, required=False, default=100, help="Print progress to screen every x iterations")
    parser.add_argument("--save_checkpoint_rate", type=int, required=False, default=1000, help="Saves progress to checkpoint files every x iterations")
    parser.add_argument("--pics_per_epoch", type=int, required=False, default=10, help="Defines the number of pictures to save each epoch.")

    return parser


class Logger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

def post_config(opt):
        args = ''
        for s in sys.argv:
            args += s + ' '
        opt.args = args
        opt.pretrained_deeplabv2_on_source_path =   {'gta5': r'./pretrained/pretrained_semseg_on_gta5.pth',
                                                     'synthia': r'./pretrained/pretrained_semseg_on_synthia.pth',
                                                     'zerowastev1': r'./pretrained/pretrained_semseg_on_zerowastev1.pth',
                                                     'synthwaste': r'./pretrained/pretrained_semseg_on_synthwaste.pth'}
        opt.folder_string = '%sGPU%d/' % (datetime.datetime.now().strftime('%d-%m-%Y::%H:%M:%S'), opt.gpus[0])
        opt.out_folder = '%s/%s' % (opt.checkpoints_dir, opt.folder_string)
        opt.src_data_list = './dataset/{}_list/'.format(opt.source)
        opt.trg_data_list='./dataset/{}_list/'.format(opt.target)

        try:
            os.makedirs(opt.out_folder)
        except OSError:
            pass

        if opt.debug_run:
            opt.print_rate = 5
            try:
                os.makedirs('./debug_runs/TrainedModels/%s' % opt.folder_string)
            except OSError:
                pass
            opt.tb_logs_dir = './debug_runs'
            opt.out_folder = './debug_runs/TrainedModels/%s' % opt.folder_string


        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)[1:-1].strip(' ').replace(" ", "")
        opt.images_per_gpu = [int(batch_size / len(opt.gpus)) for batch_size in opt.batch_size_list]
        opt.logger = Logger(os.path.join(opt.out_folder, 'log.txt'))
        sys.stdout = opt.logger
        print("Random Seed: ", opt.manual_seed)
        import torch
        opt.device = torch.device('cuda')
        random.seed(opt.manual_seed)
        torch.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed(opt.manual_seed)
        np.random.RandomState(opt.manual_seed)
        np.random.seed(opt.manual_seed)

        return opt