# SMarT-Diff: A Multi-Objective Molecular Generation Framework for Lead Optimization

> ⚠️ **Important Notice**  
> This repository accompanies a research project that **has not yet been formally published**.  
> The framework, algorithms, and model configurations are subject to change.  
> Please use with caution and cite appropriately once the work is officially released.

---

## Overview

SMarT-Diff is a **multi-objective molecular generation model** designed for lead optimization.  
It is built upon a **score-based generative modeling framework** and incorporates a **scaffold-hopping–inspired sampling strategy** to generate novel, drug-like molecules tailored to specific targets.

Key capabilities include:

- Multi-objective molecular generation (QED, SA, CNS suitability, etc.)  
- Structure-aware score modeling  
- Pharmacophore-guided sampling  
- Reinforcement learning via A2C for activity enhancement  
- Support for both 2D and 3D representation workflows

---

## Environment Setup

Create and activate the environment:

```bash
conda create -n smartdiff --file requirements.txt
conda activate smartdiff
```

## Best practices

The checkpoints provided are for reference. For optimal results, retrain the model on your specific dataset and objectives.
The visitor can download the pre-trained model from: [HuggingFace Link](https://huggingface.co/spaces/vicky963/SMarT-Diff)

## Dataset Preparation
### 1. Provided datasets

The repository includes: ZINC250k, ChEMBL

The preprocessed datasets can be downloaded from [HuggingFace Link](https://huggingface.co/spaces/vicky963/SMarT-Diff) to: ```data/```

### 2. Use your own dataset
```bash
python data/dataset_my.py
```

Modify paths in ```dataset_my.py```, then place your processed dataset under ```data/```.

### 3. Target-specific active dataset

Place your processed active compounds CSV into:
```
data/
```

SMarT-Diff will automatically load it when training or sampling.

## Model Training

Set work_type: "train" in your config and run:
```bash
python main.py
```

Ensure the configuration matches your dataset.

## Sampling Strategies
### Core basic RA Sampling

Set: ```work_type: "sample"```

Then run:
```bash
python main.py
```
### Target-Aware Generation

Prepare and convert pharmacophore input:
```bash
python data/ligand2ppgraph.py
```

SMarT-Diff will use this information during sampling.

### CNS Tasks and BBB Predictor

The BBB predictor is in: ```BBB_predictor/```

Retrain via:```BBB predictor.ipynb```


It can provide an additional reward during A2C sampling.

### A2C Policy for Activity Enhancement

Run:
```
python A2Cpolicy.py
```

The agent optimizes molecules toward higher activity using pharmacophore and BBB rewards.

## License

This project is released under the MIT License.

## Disclaimer

This is research code for an unpublished project; some components may still change.
