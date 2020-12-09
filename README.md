# Single-Image-Super-Resolution-Using-a-Self-Attention-GAN

Project of Advanced Topics in Mechatronics Systems(Deep Learning in Computer Vision)

[PROJECT REPORT](https://github.com/redlessme/Single-Image-Super-Resolution-Using-a-Self-Attention-GAN/blob/master/project_report.pdf)  
[Presentation slide](https://github.com/redlessme/Single-Image-Super-Resolution-Using-a-Self-Attention-GAN/blob/master/p8.pdf)  



## Introduction

We proposed a channel attention GAN and verified that our model outperforms state-of-the-art model SRGAN.

## Our contribution
We proposed a novel self adversarial learning architecture that leveraged the HRNet, increased estimation accuracy when occlusions and implausible poses are presented.

We designed a boundary equilibrium scheme for our adversarial training, by balancing the learning speed for our discriminator, we proved that our adversarial training strategy is more stable and can avoid mode collapse when using HRNet as the backbone.

### Architecture

![alt text](images/architecture.png)
### Channel Attention block
![alt text](images/ca.png)

## Experiment Results

### Comparision
![alt text](images/model.png)
### Ablation study
![alt text](images/ab1.png)
![alt text](images/ab2.png)
### Stablization strategies
![alt text](images/stable.png)
### Visualization of the attention maps
![alt text](images/visualization.png)
### Results
![alt text](images/r1.png)
![alt text](images/r2.png)
![alt text](images/r3.png)
### Video demo
![alt text](images/ezgif.com-gif-maker.gif)


