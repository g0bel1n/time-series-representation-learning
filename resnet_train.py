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
    "--head_type", type=str, default="regression", help="regression or classification"
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
parser.add_argument("--lr", type=float, default=None, help="learning rate")
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

parser.add_argument('--classification', type=float, default=None, help='rate for labelling')



args = parser.parse_args()
print("args:", args)
args.save_pretrained_model = f"resnet_cw{str(args.context_points)}_epochs-pretrain{str(args.n_epochs)}_model{str(args.pretrained_model_id)}"
args.save_path = "saved_models/" + args.dset + "/resnet/" + args.model_type + "/"
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def get_model(in_channels, task, len_pred, head_dropout):
    model = ResNet(in_channels, task, len_pred, head_dropout)
    print(
        "number of model params",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
    return model


set_device()

def test_func(weight_path):
    # get dataloader
    dls = get_dls(args)
    model = get_model(dls.vars, args.head_type, args.target_points, args.head_dropout).to('cuda')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
    learn = Learner(dls, model,cbs=cbs)
    out  = learn.test(dls.test, weight_path=weight_path+'.pth', scores=[acc] if args.head_type == 'classification' else [mse,mae])         # out: a list of [pred, targ, score]
    print('score:', out[2])
    # save results
    pd.DataFrame(np.array(out[2]).reshape(1,-1), columns=['mse','mae']).to_csv(args.save_path + args.save_pretrained_model + '_acc.csv', float_format='%.6f', index=False)
    return out

def find_lr():
    # get dataloader
    args.lr = 1e-4
    dls = get_dls(args)    
    model = get_model(dls.vars, args.head_type,  args.target_points, args.head_dropout)
    # get loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean') if args.head_type == 'classification' else torch.nn.MSELoss(reduction='mean')
    # get callbacks
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
        
    # define learner
    learn = Learner(dls, model, 
                        loss_func, 
                        lr=args.lr, 
                        cbs=cbs,
                        )                        
    # fit the data to the model
    suggested_lr = learn.lr_finder()
    print('suggested_lr', suggested_lr)
    return suggested_lr


def train_resnet(args):
    dls = get_dls(args)
    # get model
    model = get_model(dls.vars, args.head_type,  args.target_points, args.head_dropout)
    # get loss
    loss_func = torch.nn.CrossEntropyLoss(reduction='mean') if args.head_type == 'classification' else torch.nn.MSELoss(reduction='mean')

    cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []

    cbs += [
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
        metrics=[acc] if args.head_type == 'classification' else [mse,mae]
    )
    # fit the data to the model
    learn.fit_one_cycle(n_epochs=args.n_epochs, lr_max=args.lr, pct_start=0.2)

    train_loss = learn.recorder["train_loss"]
    valid_loss = learn.recorder["valid_loss"]
    df = pd.DataFrame(data={"train_loss": train_loss, "valid_loss": valid_loss})
    df.to_csv(
        args.save_path + args.save_pretrained_model + "_losses.csv",
        float_format="%.6f",
        index=False,
    )
  


if __name__ == "__main__":

    torch.cuda.empty_cache()
    suggested_lr = find_lr() if args.lr is None else args.lr
    args.lr = suggested_lr
    train_resnet(args)
    test_func(args.save_path+args.save_pretrained_model)
