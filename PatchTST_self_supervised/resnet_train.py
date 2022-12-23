# train resnet on custom dataset
import os
import numpy as np
import pandas as pd
import os
import torch
from torch import nn

from src.models.resnet import ResNet
from src.learner import Learner, transfer_weights
from src.callback.tracking import *
from src.callback.patch_mask import *
from src.callback.transforms import *
from src.metrics import *
from src.basics import set_device
from datautils import *


import argparse

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
# Dataset and dataloader
parser.add_argument("--dset", type=str, default="etth1", help="dataset name")
parser.add_argument("--context_points", type=int, default=512, help="sequence length")
parser.add_argument("--target_points", type=int, default=96, help="forecast horizon")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument(
    "--num_workers", type=int, default=0, help="number of workers for DataLoader"
)
parser.add_argument(
    "--scaler", type=str, default="standard", help="scale the input data"
)
parser.add_argument(
    "--features",
    type=str,
    default="M",
    help="for multivariate model or univariate model",
)
parser.add_argument(
    "--task", type=str, default="regression", help="regression or classification"
)


# RevIN
parser.add_argument(
    "--revin", type=int, default=1, help="reversible instance normalization"
)
#
parser.add_argument("--head_dropout", type=float, default=0.2, help="head dropout")
# Optimization args
parser.add_argument(
    "--n_epochs", type=int, default=10, help="number of pre-training epochs"
)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
# model id to keep track of the number of models saved
parser.add_argument(
    "--pretrained_model_id",
    type=int,
    default=1,
    help="id of the saved pretrained model",
)
parser.add_argument(
    "--model_type",
    type=str,
    default="based_model",
    help="for multivariate model or univariate model",
)


args = parser.parse_args()
print("args:", args)
args.save_pretrained_model = f"resnet_cw{str(args.context_points)}_epochs-pretrain{str(args.n_epochs)}_model{str(args.pretrained_model_id)}"
args.save_path = "saved_models/" + args.dset + "/resnet/" + args.model_type + "/"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def get_model(in_channels, task, n_classes, head_dropout):
    model = ResNet(in_channels, task, n_classes, head_dropout)
    print(
        "number of model params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


set_device()


def train_resnet(args):
    dls = get_dls(args)
    # get model
    model = get_model(dls.vars, args.task, 2, args.head_dropout)
    # get loss
    loss_func = torch.nn.MSELoss(reduction="mean")

    cbs = [
        SaveModelCB(
            monitor="valid_loss", fname=args.save_pretrained_model, path=args.save_path
        )
    ]
    # define learner
    learn = Learner(
        dls,
        model,
        loss_func,
        lr=args.lr,
        cbs=cbs,
        # metrics=[mse]
    )
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=args.lr)

    train_loss = learn.recorder["train_loss"]
    valid_loss = learn.recorder["valid_loss"]
    df = pd.DataFrame(data={"train_loss": train_loss, "valid_loss": valid_loss})
    df.to_csv(
        args.save_path + args.save_pretrained_model + "_losses.csv",
        float_format="%.6f",
        index=False,
    )


if __name__ == "__main__":

    train_resnet(args)
