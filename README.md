
PyTorch implementation for [Robust Object Re-identification with Coupled Noisy Labels](http://pengxi.me/wp-content/uploads/2024/01/Manuscript.pdf) (IJCV 2024).

LCNL extends the previous work [Learning with Twin Noisy Labels for Visible-Infrared Person Re-Identification](https://openaccess.thecvf.com/content/CVPR2022/papers/Yang_Learning_With_Twin_Noisy_Labels_for_Visible-Infrared_Person_Re-Identification_CVPR_2022_paper.pdf) (CVPR 2022) by generalizing DART to both single- and cross-modality ReID tasks with improved loss function.


## Introduction

### LCNL framework
<img src="https://github.com/XLearning-SCU/2022-CVPR-DART/blob/main/figs/framework.png"  width="760" height="268" />

## Requirements

- Python 3.7
- PyTorch ~1.7.1
- numpy
- scikit-learn
## Datasets

### SYSU-MM01 and RegDB
We follow [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ) to obtain the datasets.

### Market1501, Duke-MTMC and VeRi-776
We follow [TransReID](https://github.com/damo-cv/TransReID) to obtain the datasets.

## Visible-infrared ReID

### Training

Modify the ```data_path``` and  specify the ```noise_ratio``` to train the model.

```train
# SYSU-MM01: noise_ratio = {0, 0.2, 0.5}
python run.py --gpu 0 --dataset sysu --data-path data_path --noise-rate 0. --savename sysu_lcnl_nr0 --op-type weighty

# RegDB: noise_ratio = {0, 0.2, 0.5}, trial = 1-10
python run.py --gpu 0 --dataset regdb --data-path data_path --noise-rate 0. --savename regdb_lcnl_nr0 --trial 1
```
### Evaluation

Modify the  ```data_path``` and ```model_path``` to evaluate the trained model. 

```
# SYSU-MM01: mode = {all, indoor}
python test.py --gpu 0 --dataset sysu --data-path data-path --model_path model_path --resume-net1 'sysu_lcnl_nr0_net1.t' --resume-net2 'sysu_lcnl_nr0_net1.t' --mode all

# RegDB: --tvsearch or not (whether thermal to visible search)
python test.py --gpu 0 --dataset regdb --data-path data-path --model_path model_path --resume-net1 'regdb_lcnl_nr20_trial{}_net1.t' --resume-net2 'regdb_lcnl_nr20_trial{}_net2.t'
```

### Reproduce
We provide the [checkpoints](https://pan.baidu.com/s/1SqYBxbxXIj_4yYZOWhlDgA?pwd=es2p) (for evaluation) and [noise indexes](https://pan.baidu.com/s/1d8McuazzrYocRCNnKE4vjg?pwd=aam5) (should be placed on the dataset path for training) for result reproducing. 
<!-- Note that the reproduced results would be slightly different from the results in the paper due to the code reshaping.  -->

## Todo

- Release the code for vehicle ReID task.
- Release the code for visible ReID task.



## Citation

If LCNL is useful for your research, please consider citing:
```
@article{yang2024lcnl,
  title={Robust Object Re-identification with Coupled Noisy Labels},
  author={Yang, Mouxing and Huang, Zhenyu and Peng, Xi},
  journal={International Journal of Computer Vision},
  year={2024},
  publisher={Springer}
}
```
or the previous conference version:
```
@InProceedings{Yang_2022_CVPR,
    author={Yang, Mouxing and Huang, Zhenyu and Hu, Peng and Li, Taihao and Lv, Jiancheng and Peng, Xi},
    title={Learning With Twin Noisy Labels for Visible-Infrared Person Re-Identification},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={14308-14317}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

## Acknowledgements
The code is based on [ADP](https://github.com/mangye16/Cross-Modal-Re-ID-baseline/tree/master/ICCV21_CAJ), [TransReID](https://github.com/damo-cv/TransReID) and [DART](https://github.com/XLearning-SCU/2022-CVPR-DART) licensed under Apache 2.0.