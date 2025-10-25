# MMPD-DTA
We propose a multi-modal deep learning model for predicting drug-target binding affinity
```
conda env create -f environment.yml -n MMPD_DTA
```
 
## Data
We use PDBBind2020 as our dataset，http://pdbbind.org.cn/
## Train
If you want to train the model, run python [main.py](main.py). Our training set at [google](https://drive.google.com/file/d/1Ny3yU0H89-q47ahjtLw8cuA4adiyzLks/view?usp=drive_link.)
##
We provide test scripts to reproduce the data in the paper.[test.py](test.py)
## Cite
If our work is helpful to you, we encourage you to cite our paper:https://doi.org/10.1021/acs.jcim.4c01528
```
@article{wang2025mmpd,
  title={MMPD-DTA: Integrating Multi-Modal Deep Learning with Pocket-Drug Graphs for Drug-Target Binding Affinity Prediction},
  author={Wang, Guishen and Zhang, Hangchen and Shao, Mengting and Sun, Shisen and Cao, Chen},
  journal={Journal of Chemical Information and Modeling},
  volume={65},
  number={3},
  pages={1615--1630},
  year={2025},
  publisher={ACS Publications}
}
```
