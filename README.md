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

````
python patchtst_pretrain.py --dset_pretrain ettm1 --mask_ratio .4 --stride 8 --patch_len 16 --context_points 336 --n_epochs_pretrain 100 --batch_size 128
```