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
BUILD DATALOADER FOR GUNPOINT