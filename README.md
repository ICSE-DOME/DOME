# ICSE22_DOME
This is a replication package for `On-Demand Code Comment Generation`.<br>
Our project is public at: <https://github.com/ICSE-DOME/DOME>

## Content
1. [Project Summary](#1-Project-Summary)<br>
2. [Get Started](#2-Get-Started)<br>
&ensp;&ensp;[2.1 Requirements](#21-Requirements)<br>
&ensp;&ensp;[2.2 Dataset](#22-Dataset)<br>
&ensp;&ensp;[2.3 Comment Intent Labeling Tool](#23-Comment-Intent-Labeling-Tool)<br>
&ensp;&ensp;[2.4 Code Comment Generator](#24-Code-Comment-Generator)<br>

## 1 Project Summary
Existing automatic code comment generators mainly focus on producing a general description of functionality for a given code snippet without considering developer intentions. However, in real-world practice, comments are complicated, which often contain information reflecting various intentions of developers, e.g.,  functionality summarization, design rationale, implementation details, code properties, etc. 
To bridge the gap between automatic code comment generation and real-world comment practice, we define On-Demand Code Comment Generation, which can generate intent-aware comments for the same source code with different intents. 
To tackle this challenging task, we propose DOME, an approach that utilizes Intent-guided Selective Attention to explicitly select intent-relevant information from the source code, and produces various comments reflecting different intents. 
Our approach is evaluated on two real-world Java datasets, and the experimental results show that our approach outperforms the state-of-the-art baselines. 


## 2 Get Started
### 2.1 Requirements
* Hardwares: NVIDIA GeForce RTX 3060 GPU, intel core i5 CPU
* OS: Ubuntu 20.04
* Packages: 
  * python 3.9
  * pytorch 1.9.0
  * cuda 11.1
  * transformers 4.9.2
  * numpy
  * tqdm

### 2.2 Dataset
#### 2.2.1 benchmark datasets
DOME is evaluated on [Funcom](http://leclair.tech/data/funcom/) and [TLC](https://github.com/xing-hu/TL-CodeSum) benchmark datasets.<br>
[Shi et al. FSE22](https://arxiv.org/pdf/2207.05579) have reported that many benchmark datasets have noisy data, e.g. *Verbose Sentence*, *Content Tampering*. So we directly use the "clean" version of the two datasets open sourced by them (https://github.com/BuiltOntheRock/FSE22_BuiltOntheRock).
#### 2.2.2 manually labeled dataset
To train our comment-intent labeling tool, we randomly sample 20K code-comment pairs from Funcom and TLC (10K data for each), and invite five developers to manually classify the data into six intent categories. The intent-labeled dataset can be found in ```src/comment_classifier/dataset/```.

### 2.3 Comment Intent Labeling Tool
1. Download the pretrained CodeBERT model from [transformers](https://huggingface.co/microsoft/codebert-base) and put the model file into the ```src/comment_classifier``` directory.

2. Train the Comment-Intent Labeling Tool *COIN* model with intent-labeled dataset.
```
python src/comment_classifier/train.py
```
3. Use the well-trained COIN to annotate the two benchmark dataset automatically.
```
python src/comment_classifier/prediction.py
```

### 2.4 Code Comment Generator
1. Go the ```src/comment_generator``` directory, process the datasets labeled by COIN tool and generate the vocabulary:
```
cd src/comment_generator
python dataloader.py
```
2. Train DOME model:
```
python train.py
```
3. Test DOME model:
```
python prediction.py
```
