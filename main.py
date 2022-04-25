import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from argument_parser import parse_args
from dataloader import Dataloader
from model_v1 import Model
from train_test_v1 import train_test_v1
from utils import seed_setting, init_logger, get_summary_writer, record_configuration


def main(args):
    seed = args.random_seed
    seed_setting(seed)
    # dataset information
    data_dir = args.data_dir
    data_type = args.data_type
    train_test_ratio = args.train_test_ratio
    dataloader = Dataloader(data_dir, data_type, label_filepath='D:\\PyCharmProjects\\gnn_pd\\data\\y_label.txt')
    dataset_config = {
        'data_dir': data_dir,
        'data_type': data_type,
        'train_test_ratio': train_test_ratio,
    }

    train_dataloader, test_dataloader = dataloader.generate_dataloader(train_test_ratio)

    # model information
    model_config = {
        'in_feats': args.in_feats,
        'n_hidden': args.n_hidden,
        'n_layers': args.n_layers,
        'activation': F.relu,
        'dropout': args.dropout,
    }
    model = Model(model_config)

    # training information
    n_epochs = args.n_epochs
    lr = args.lr
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, lr=lr)
    loss_func = nn.HuberLoss(reduction='mean', delta=1.0)
    # loss_func = nn.MSELoss()
    train_config = {
        'n_epoch': n_epochs,
        'lr': lr,
        'log_filepath': 'runs',
        'random_seed': args.random_seed,
        'device': args.device,
        'comment': args.comment,
    }
    summary_writer, log_dir = get_summary_writer(train_config['log_filepath'])
    logger = init_logger(os.path.join(log_dir, 'logging.txt'))
    record_configuration(save_dir=log_dir, configuration_dict={
        'MODEL': model_config,
        'DATASET': dataset_config,
        'TRAIN': train_config,
    })
    train_test_v1(train_dataloader, test_dataloader,
                  model, optimizer, loss_func,
                  n_epochs, summary_writer, log_dir, logger)


if __name__ == '__main__':
    args = parse_args()
    main(args)
