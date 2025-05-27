# Linear_Reasoning_Features (ACL 2025 Findings)


This repository contains the data and code for the experiments in our paper titled **[The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction]**

* **Arxiv:** https://arxiv.org/abs/2503.23084


**1**
<p align="center">
  <img src="https://github.com/yihuaihong/ConceptVectors.github.io/blob/main/static/images/interp_reasoning.png" width="1000"></a>
  <br />
  <em>Illustration of Linear Reasoning Features(LiReFs)</em>
</p>


## Quick Links
- [Linear Reasoning Features](#lirefs)
  - [Overview](#overview)
  - [How to Run](#how-to-run)
  - [How to Cite](#how-to-cite)

## Overview
You can reproduce the experiments in our paper.

> **Abstract**
> Large language models (LLMs) excel on a variety of reasoning benchmarks, but previous studies suggest they sometimes struggle to generalize to unseen questions, potentially due to over-reliance on memorized training examples. However, the precise conditions under which LLMs switch between reasoning and memorization during text generation remain unclear. In this work, we provide a mechanistic understanding of LLMs' reasoning-memorization dynamics by identifying a set of linear features in the model's residual stream that govern the balance between genuine reasoning and memory recall. These features not only distinguish reasoning tasks from memory-intensive ones but can also be manipulated to causally influence model performance on reasoning tasks. Additionally, we show that intervening in these reasoning features helps the model more accurately activate the most relevant problem-solving capabilities during answer generation. Our findings offer new insights into the underlying mechanisms of reasoning and memory in LLMs and pave the way for the development of more robust and interpretable generative AI systems. To support this, we release our code at https://github.com/yihuaihong/Linear_Reasoning_Memory_Features.


## How to Run

**Step1: unzip the dataset** 
```sh
unzip dataset.zip
```

**Step2: Storing the Hidden-states of Models on Certain Tasks**
Please run ./reasoning_representation/LiReFs_storing_hs.ipynb

**Step3: Create the PCA and other Figures**
Please run ./reasoning_representation/Figures_Interp_Reason&Memory.ipynb

**Step4: Intervention Experiments**
```sh
cd Intervention
python features_intervention.py
```


## How to Cite
```
@misc{hong2025reasoningmemorizationinterplaylanguagemodels,
      title={The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction}, 
      author={Yihuai Hong and Dian Zhou and Meng Cao and Lei Yu and Zhijing Jin},
      year={2025},
      eprint={2503.23084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.23084}, 
}
```

