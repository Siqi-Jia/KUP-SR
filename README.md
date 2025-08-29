# KUPSR

This is the official implementation for *Sequential Recommendation via Knowledge Graph-Enhanced Multi-Relational Learning and Temporal-Aware User Preference Modeling*.

## Introduction

KUPSR (**Knowledge-enhanced User Preference Sequential Recommender Model**) is a unified framework for sequential recommendation that jointly models:

- **Knowledge Graph-Enhanced Multi-Relational Learning**: constructs a multi-relational item-item knowledge graph and integrates relation strengths with a Fourier-based temporal decay function.  
- **User Neighborhood Enhancement**: aggregates embeddings of similar users via attention pooling to capture collaborative signals and latent preferences.  
- **Sequential Modeling**: employs a self-attention mechanism combined with Bi-LSTM to capture both short-term fluctuations and long-term evolution of user behaviors.  

Extensive experiments on multiple real-world datasets demonstrate that KUPSR outperforms state-of-the-art methods in accuracy, robustness, and generalization.

## Getting Started

1. Install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) with Python >= 3.7  
2. Clone the repository and install requirements

```bash
git clone https://github.com/Siqi-Jia/KUP-SR.git
```

3. Install requirements and step into the `src` folder

```bash
cd KUPSR-main
pip install -r requirements.txt
cd src
```

4. Run model on the built-in dataset

```bash
# Example: KUPSR on Office dataset
python main.py --model_name KUPSR --emb_size 64 --lr 1e-4 --l2 1e-6 --num_heads 4 --num_layers 5 --history_max 20 --dataset Office --epoch 200 --gpu 0
```
