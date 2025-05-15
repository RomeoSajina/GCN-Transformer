# GCN-Transformer
Official repository for paper titled "GCN-Transformer: Multi-task Graph Convolutional Network and Transformer for Multi-person Pose Forecasting"

> Multi-person pose forecasting involves predicting the future body poses of multiple individuals over time, involving complex movement dynamics and interaction dependencies. Its relevance spans various fields, including computer vision, robotics, human-computer interaction, and surveillance. This paper introduces GCN-Transformer, a novel model for multi-person pose forecasting that leverages the integration of Graph Convolutional Network and Transformer architectures. We integrated novel loss terms during the training phase to incentivize the model to learn both interaction dependencies and joint trajectory. Additionally, we conducted an ablation study to analyze the effects of various model components. Through comprehensive evaluations on the SoMoF Benchmark and ExPI datasets, employing VIM and MPJPE, our findings consistently highlight the superior performance of GCN-Transformer over existing models. Furthermore, we introduce a novel pose forecasting evaluation metric called Final Joint Position And Trajectory Error (FJPTE), which comprehensively assesses both local movement dynamics and global movement errors by considering the final position and the trajectory leading up to it. Findings on all evaluation metrics underscore the potential of GCN-Transformer to advance multi-person pose forecasting, offering promising applications in diverse domains.

The code for this paper will be made available soon.

For inquiries, please contact rsajina@unipu.hr

## Getting Started

Clone the repo:

```
git clone https://github.com/RomeoSajina/GCN-Transformer.git
```

(Optional) Create a Conda environment:
```
conda create -n gcn-transformer python=3.8
```

Install the requirements using `pip`:
```
pip install -r requirements.txt
```

### Requirements

- torch==1.13.1
- numpy==1.24.4
- scipy==1.10.0
- einops==0.6.1


## Data

First, create a `data/` folder in the repo root directory. We expect the following structure:
```
data/
    3dpw/
        sequenceFiles/
            test/
            train/
            validation/
    somof_data_3dpw/
        3dpw_test_frames_in.json
        3dpw_test_in.json
        ...
    amass/
        BioMotionLab_NTroje/
        BMLmovi/
        CMU/
        smpl_skeleton.npz
    expi/
        ExPI_mocap_data/
            acro1/
                ...
            acro2/
                ...
```

Datasets links:
- [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/license.html)
- [SoMoF](https://virtualhumans.mpi-inf.mpg.de/3DPW/licensesmof.html)
- [AMASS](https://amass.is.tue.mpg.de/)
- [ExPI](https://team.inria.fr/robotlearn/research/expi-dataset/)



## Setup and all experiments
Checkout `Run.ipynb` notebook for all experiments, and `Run_ablation.ipynb` for ablation study


## Training

The model is trained with the training script `train.py`:
```
python train.py --dataset 3dpw
```


## Evaluation
We provide a script to evaluate trained GCN-Transformer models. You can run
```
python test.py --dataset 3dpw --ckp ./logs/3dpw/best_epoch.pt
```
to get these metrics.


### Citing
If you use our code, please cite our work

```
@Article{sajina25103136,
AUTHOR = {Šajina, Romeo and Oreški, Goran and Ivašić-Kos, Marina},
TITLE = {GCN-Transformer: Graph Convolutional Network and Transformer for Multi-Person Pose Forecasting Using Sensor-Based Motion Data},
JOURNAL = {Sensors},
VOLUME = {25},
YEAR = {2025},
NUMBER = {10},
ARTICLE-NUMBER = {3136},
URL = {https://www.mdpi.com/1424-8220/25/10/3136},
ISSN = {1424-8220},
DOI = {10.3390/s25103136}
}
```
