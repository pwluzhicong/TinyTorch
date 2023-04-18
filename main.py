import argparse
import numpy as np

from train_utils import train_func
from tinytorch import Tensor, MLP, Parameter, MomentumOptimizer, Linear, Sigmoid, square_loss, log_binary_loss
from dataset import DataLoader


arg_parser = argparse.ArgumentParser(description='Train and test demo of MLP')


def arguments():
    arg_parser.add_argument("--input_data", dest="input_data", required=True)
    arg_parser.add_argument("--num_units", dest="num_units", required=True)
    arg_parser.add_argument("--loss_func", dest="loss_func",choices=["square_loss", "log_binary_loss"], required=True)
    # type=lambda x :[int(y) for y in x.split(",")])
    arg_parser.add_argument("--activation_func", choices=["sigmoid", "tanh"], dest="activation_func", default="sigmoid")
    arg_parser.add_argument("--batch_size", dest="batch_size", type=int,default=16)
    arg_parser.add_argument("--num_epochs", dest="num_epochs", type=int, default=50)
    arg_parser.add_argument("--learning_rate", dest="learning_rate", type=float, default=1e-2)
    arg_parser.add_argument("--momentum", dest="momentum", type=float, default=0.9)
    arg_parser.add_argument("--l2_norm", dest="l2_norm", type=float, default=1e-3)
    arg_parser.add_argument("--test_ratio", dest="test_ratio", type=float, default=0.2)
    arg_parser.add_argument("--plot_to_file", dest="plot_to_file", type=str, default=None)

def main():
    arguments()
    args = arg_parser.parse_args()
    print(f"Experiments args: {vars(args)}")
    train_func(**vars(args))
    

if __name__ == "__main__":
    main()

