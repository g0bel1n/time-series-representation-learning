Time Series and representation learning : CNN vs Transformers


Code for PatchTST is from https://github.com/yuqinie98/PatchTST.


| Trainings                | pretrain on | finetuned on | done |
| ------------------------ | ----------- | ------------ | ---- |
| baseline Transformer     | ettm1       | ettm1        |      |
| financial TS Transformer | ive         | ive          |      |


### ToDo List 

- Train Patcht TST on 
 - [ ] ettm1, ive and GunPoint
  - [ ] IVE
  - [ ] Gunpoint
- Train a MCNN on:
  - [ ] ettm1, ive and GunPoint
  - [ ] IVE
  - [ ] Gunpoint
- Look at for GunPoint
  - [ ] CAM
  - [ ] MDS
  - [ ] Attention Map 
- [ ] Same for IVE-classification

### Command lines

for PachTST

```
python patchtst_pretrain.py --dset_pretrain ettm1 --mask_ratio .4 --stride 8 --patch_len 16 --context_points 336 --n_epochs_pretrain 100 --batch_size 128
````

For resnet
```
python resnet_train.py --dset ettm1  --context_points 336 --n_epochs 100 --batch_size 256 --lr 0.01
```

```
python resnet_train.py --dset gunpoint --batch_size 8  --head_type classification  --context_points 150 --target_points 2  --revin 0 --n_epochs 20
```

```
python patchtst_supervised.py --dset gunpoint --batch_size 8 --patch_len 16 --stride 8 --head_type classification --features U --context_points 150 --target_points=2 --revin 0 --n_epochs 10  --is_train 1
```


```
python patchtst_finetune.py --dset_finetune ettm1 --patch_len 16 --stride 8 --batch_size 256 --context_points 336  --features M --target_points 2 --pretrained_model saved_models/ettm1/patchtst/patchtst_pretrained_cw336_patch16_stride8_epochs-pretrain100_mask0.4_model1.pth --is_finetune 1 --revin 0 --head_type classification --classification .05
```

```
python resnet_train.py --dset gunpoint  --batch_size 32  --head_type classification  --context_points 150 --target_points 2  --revin 0 --n_epochs 20 --classification .05
```
