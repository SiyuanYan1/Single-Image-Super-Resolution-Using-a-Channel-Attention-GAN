# Single-Image-Super-Resolution-Using-a-Self-Attention-GAN

Project of ENGN8535(Advanced Topics in Mechatronics Systems)

[PROJECT REPORT](https://github.com/redlessme/Single-Image-Super-Resolution-Using-a-Self-Attention-GAN/blob/master/project_report.pdf)  
[Presentation slide](https://github.com/redlessme/Single-Image-Super-Resolution-Using-a-Self-Attention-GAN/blob/master/p8.pdf)  



## Introduction
Since our code is developed on the official implementation of [*Deep High-Resolution Representation Learning for Human Pose Estimation*](https://arxiv.org/abs/1902.09212). This instruction is highly based and similar to the original [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

## Our contribution
We proposed a novel self adversarial learning architecture that leveraged the HRNet, increased estimation accuracy when occlusions and implausible poses are presented.

We designed a boundary equilibrium scheme for our adversarial training, by balancing the learning speed for our discriminator, we proved that our adversarial training strategy is more stable and can avoid mode collapse when using HRNet as the backbone.
## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 1 NVIDIA RTX 2080Ti and 4 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)** (We use pytorch == 1.6.0)
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

5. Download pretrained models from our model zoo([GoogleDrive](https://drive.google.com/file/d/1irp5Ozn2qf5DYqRkqJSjttidx6YjZfXP/view?usp=sharing))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            |-- hrgan_coco
            |   |-- hrgan_w32_256x192.pth
            |-- hrgan_mpii
            |   |-- hrgan_w32_256x256.pth
   ```
   
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. The authors of HRNet have converted them into json format, you also need to download them from [GoogleDrive](https://drive.google.com/file/d/1irp5Ozn2qf5DYqRkqJSjttidx6YjZfXP/view?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. Person detection result of COCO val2017 and test-dev2017 are also provided to reproduce our multi-person pose estimation results. Please download from [GoogleDrive](https://drive.google.com/file/d/1irp5Ozn2qf5DYqRkqJSjttidx6YjZfXP/view?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on MPII dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/hrgan_mpii/hrgan_w32_256x256.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/hrgan/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset using model zoo's models([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
 

```
python tools/test.py \
    --cfg experiments/coco/hrgan/w32_256x192_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/hrgan_coco/hrgan_w32_256x192.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/hrgan/w32_256x192_adam_lr1e-3.yaml \
```

### Visualization

#### Visualizing predictions on MPII validate set

```
python visualization/plot_mpii.py \
    TEST.MODEL_FILE models/pytorch/hrgan_coco/hrgan_w32_256x192.pth
```

<img src="figures\visualization\mpii\hrgan\000059794.jpg" height="215"><img src="figures\visualization\mpii\hrgan\016035911.jpg" height="215"><img src="figures\visualization\mpii\hrgan\016047648.jpg" height="215">

### Citation
If you use the code or models in your research, please cite with the following papers as the original [github page](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) requires:
```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```
