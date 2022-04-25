import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="gnn pd arguments")

    # dataset configuration
    parser.add_argument('--data_dir', type=str, default='D:\\PyCharmProjects\\gnn_pd\\data\\graph')
    parser.add_argument('--data_type', type=str, default='DFC')

    # model configuration
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--in_feats', type=int, default=246)
    parser.add_argument('--n_hidden', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    # training configuration
    parser.add_argument('--n_epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--log_filepath', type=str, default='run')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train_test_ratio', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=1024, help='random seed for training and testing')
    parser.add_argument('--comment', type=str, default='default comment', help='comment for each experiment')
    args = parser.parse_args()
    return args


