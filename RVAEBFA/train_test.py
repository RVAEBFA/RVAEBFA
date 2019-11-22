import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from utils import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import datetime
input_dict= {'kddcup': 118}
# input_dict = {'arrhythmia':274}
# input_dict = {'kddcup': 118, 'arrhythmia':274, 'mnist':100, 'musk':166, 'optdigits':64, 'pendigits':16, 'satimage':36, 'speech':400, 'MNIST':784}
# input_dict = {'arrhythmia':274, 'musk':166, 'optdigits':64, 'pendigits':16, 'satimage':36, 'speech':400, 'MNIST':784}
latent_dict = {'kddcup': 15, 'arrhythmia':17, 'mnist':10, 'musk':10, 'optdigits':10, 'pendigits':5, 'satimage':10, 'speech':10, 'MNIST':10}

def str2bool(v):
    return v.lower() in ('true')

def main(config, name):
    # For fast training
    cudnn.benchmark = True
    # Create directories if not exist
    mkdir(config.log_path)
    mkdir(config.model_save_path)
    train_data_loader = get_loader(config.data_path, batch_size=config.batch_size, mode=config.mode, train_contaminate_ratio=0.8)
    test_data_loader = get_loader(config.data_path, batch_size=128, mode='test', test_P_N_ratio=2)
    # Solver

    
    input_dim = input_dict[name]
    latent_dim = latent_dict[name]
    bfa_range = np.arange(0, 1.01, 0.1)
    metric = []
    for bfa in bfa_range:
        solver = Solver(train_data_loader, test_data_loader, vars(config), input_dim, latent_dim, name,BFA=bfa)
        if config.mode == 'train':
            result = solver.train_total()
            result = np.mean(result, axis=0)
            metric.append(result)
    return metric, bfa_range


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model hyper-parameters
    parser.add_argument('--lr', type=float, default=1e-3)
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lambda_energy', type=float, default=0.1)
    parser.add_argument('--lambda_cov_diag', type=float, default=0.0005)
    parser.add_argument('--pretrained_model', type=str, default=False)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default='False')

    # Path
    parser.add_argument('--data_path', type=str, default='../DATA/satimage.npz')
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--model_save_path', type=str, default='./models')

    # Step size
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=194)
    parser.add_argument('--model_save_step', type=int, default=194)

    config = parser.parse_args()
 
    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for name in sorted(input_dict.keys()):        
        config.data_path = '../DATA/' + name + '.npz'
        metric, contaminate_range = main(config, name)
        np.save('./result/' + currentTime + '-BFA-' + name + '.npy', {'metric':metric, 'range':contaminate_range})
