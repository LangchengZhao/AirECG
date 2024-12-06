# AirECG
IMWUT/Ubicomp 2024

AirECG: Contactless Electrocardiogram for Cardiac Disease Monitoring via mmWave Sensing and Cross-domain Diffusion Model

## Overview
The electrocardiogram (ECG) has always served as a crucial biomedical examination for cardiac diseases monitoring and diagnosing. Typical ECG measurement requires attaching electrodes to the body, which is inconvenient for long-term monitoring. Recent wireless sensing maps wireless signals reflected from human chest into electrical activities of heart so as to reconstruct ECG contactlessly. While making great progress, we find existing works are effective only for healthy populations with normal ECG, but fall short when confronted with the most desired usage scenario: reconstructing ECG accurately for people with cardiac diseases such as atrial fibrillation, premature ventricular beat. To bridge the gap, we propose AirECG, which moves forward to reconstruct ECG for both healthy people and even cardiac patients  with morbid ECG, i.e., irregular rhythm and anomalous ECG waveform, via contactless millimeter-wave sensing. To realize AirECG, we first custom-design a cross-domain diffusion model that can perform multiple iteration denoising inference, in contrast with the single-step generative models widely used in previous works. In this way, AirECG is able to identify and eliminate the distortion due to the unstable and irregular cardiac activities, 
so as to synthesize ECG even during abnormal beats. Furthermore, we enhance the determinacy of AirECG, i.e., to generate high-fidelity ECG, by designing a calibration guidance mechanism to combat the inherent randomness issue 
of the probabilistic diffusion model. Empirical evaluation demonstrates AirECG's ability of ECG synthesis with Pearson correlation coefficient (PCC) of 0.955 for normal beats. Especially for abnormal beats, the PCC still exhibits a strong correlation of 0.860, with 15.0\%~21.1\% improvement compared with state-of-the-art approaches.

We have released the training and inference source code related to the Cross-domain diffusion model. Developers can configure the training set and test set according to their needs to train a generative model for electrocardiograms (ECG) and perform inference. In the code examples, random values are assigned to the training and testing datasets to ensure that the training and inference processes can run without error. For further development, we recommend using millimeter wave and ECG data with the same shape for training and inference.

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/LangchengZhao/AirECG.git
cd AirECG
```

Then create a Conda environment. 

```bash
conda env create -f environment.yml
conda activate AirECG
```

## Training AirECG
If you want to train AirECG on real datasets, the DataLoader（line 103） in `train.py` should be modified. The random value in the following code should be replaced by mmWave and ECG data.

```Python
from torch.utils.data import DataLoader,Dataset

def DataLoader_example(bs):
    #Load your data here
    train_mmwave = torch.randn(96, 8, 1024)
    train_ecg = torch.randn(96, 1024)

    test_mmwave = torch.randn(96, 8, 1024)
    test_ecg = torch.randn(96, 1024)

    ref_ecg = torch.randn(96, 1024)
#The remaining content has been omitted.
```


Then, you can run the `train.sh` to start up the model training. 
```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 --master_port='29500' train.py --global-batch-size 96
```

## Inference
Also for model inference, the DataLoader（line 52） in `inference.py` should be modified. Besides, the model weights path should also be specified when run the Python file, like the following:

```bash
python inference.py --ckpt /path/of/your_model_weight
```

## BibTeX

```bibtex
@article{zhao2024airecg,
  title={AirECG: Contactless Electrocardiogram for Cardiac Disease Monitoring via mmWave Sensing and Cross-domain Diffusion Model},
  author={Zhao, Langcheng and Lyu, Rui and Lei, Hang and Lin, Qi and Zhou, Anfu and Ma, Huadong and Wang, Jingjia and Meng, Xiangbin and Shao, Chunli and Tang, Yida and others},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={8},
  number={3},
  pages={1--27},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgments
This work was supported in part by Beijing Natural Science Foundation, the Innovation Research Group Project of NSFC, the Youth Top Talent Support Program.
The codebase borrows from OpenAI's ADM and Meta's DiT repos, most notably [ADM](https://github.com/openai/guided-diffusion), [DiT](https://github.com/facebookresearch/DiT).