# ICSE22_DOME
This is a replication package for `On-Demand Code Comment Generation`. 
Our project is public at: <https://github.com/ICSE-DOME/DOME>

## Content
1. [Get Started](#1-Get-Started)<br>
&ensp;&ensp;[1.1 Requirements](#11-Requirements)<br>
&ensp;&ensp;[1.2 Dataset](#12-Dataset)<br>
&ensp;&ensp;[1.3 Train and Test](#13-Train-and-Test)<br>
2. [Project Summary](#2-Project-Summary)<br>

## 1 Get Started
### 1.1 Requirements
* Hardwares: NVIDIA GeForce RTX 3060 GPU, intel core i5 CPU
* OS: Ubuntu 20.04
* Packages: 
  * python 3.9
  * pytorch 1.9.0
  * cuda 11.1
  * transformers 4.9.2
  * numpy
  * tqdm

### 1.2 Dataset
DOME is evaluated on Funcom and TLC benchmark datasets. We directly use the two datasets open sourced by the [previous work (FSE2022)](https://github.com/BuiltOntheRock/FSE22_BuiltOntheRock).

### 1.3 Train and Test


## 2 Project Summary
Existing automatic code comment generators mainly focus on producing a general description of functionality for a given code snippet without considering developer intentions. However, in real-world practice, comments are complicated, which often contain information reflecting various intentions of developers, e.g.,  functionality summarization, design rationale, implementation details, code properties, etc. 
To bridge the gap between automatic code comment generation and real-world comment practice, we define On-Demand Code Comment Generation, which can generate intent-aware comments for the same source code with different intents. 
To tackle this challenging task, we propose DOME, an approach that utilizes Intent-guided Selective Attention to explicitly select intent-relevant information from the source code, and produces various comments reflecting different intents. 
Our approach is evaluated on two real-world Java datasets, and the experimental results show that our approach outperforms the state-of-the-art baselines. 
