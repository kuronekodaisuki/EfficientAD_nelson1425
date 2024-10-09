import argparse
import os
import torch
import yaml

from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from efficientad import test


# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

def main(config):
        
    if config.dataset == 'mvtec_ad':
        dataset_path = config.mvtec_ad_path
    elif config.dataset == 'mvtec_loco':
        dataset_path = config.mvtec_loco_path

    train_dir = os.path.join(config.output_dir, 'trainings',
                                    config.dataset, config.subdataset)
    test_dir = os.path.join(config.output_dir, 'anomaly_maps',
                                   config.dataset, config.subdataset, 'test')
    print(train_dir, test_dir)

    teacher = torch.load(os.path.join(train_dir, 'teacher_final.pth'))
    student = torch.load(os.path.join(train_dir, 'student_final.pth'))
    autoencoder = torch.load(os.path.join(train_dir, 'autoencoder_final.pth'))
    t_mean, t_std, g_std_s, g_std_e, g_ae_s, g_ae_e = LoadParameters(train_dir)

# Load parameters for Inference
def LoadParameters(file_path):
    teacher_mean = torch.load(os.path.join(file_path, 'teacher_mean.pt'))
    teacher_std = torch.load(os.path.join(file_path, 'teacher_std.pt'))
    q_st_start = torch.load(os.path.join(file_path, 'q_st_start.pt'))    
    q_st_end = torch.load(os.path.join(file_path, 'q_st_end.pt'))    
    q_ae_start = torch.load(os.path.join(file_path, 'q_ae_start.pt'))    
    q_ae_end = torch.load(os.path.join(file_path, 'q_ae_end.pt'))    
    return teacher_mean, teacher_std, q_st_start, q_st_end, q_ae_start, q_ae_end
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_loco',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='breakfast_box',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)

    main(parser.parse_args())