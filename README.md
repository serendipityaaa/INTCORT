<div align="center">

# INTCORT

### Training-Free Spatial Reasoning Enhancement for Vision-Language Models via Input Transformations and Confidence Routing

**Haoran Sun<sup>1</sup>**, **Jingqi Xu<sup>2</sup>**, **Yanhui Li<sup>3</sup>**, **Enci Liu<sup>4</sup>**, **Kaidi Xu<sup>5</sup>**, **Yanwei Liu<sup>6</sup>**

University of Hong Kong · University of Southern California · China Telecom  
Columbia University · City University of Hong Kong · Chinese Academy of Sciences

</div>

<p align="center">
  <img src="assets/overview.png" width="100%" alt="Overview of INTCORT">
</p>

## Overview

Vision-Language Models (VLMs) still struggle with basic spatial reasoning questions such as left/right and above/below. INTCORT is a training-free spatial reasoning enhancement framework that constructs multiple inference views through input transformations and aggregates their predictions via relation-token confidence routing.

INTCORT does not fine-tune the VLM or modify its internal mechanisms. It uses a lightweight language model to screen spatial queries, applies geometric image transformations or query-direction reversal, maps transformed predictions back to the original relation space, and produces the final spatial prediction through confidence-weighted routing.

This repository provides the Qwen2-VL-7B implementation used in the paper.

## Installation

```bash
git clone https://github.com/serendipityaaa/INTCORT.git
cd INTCORT

conda create -n intcort python=3.10 -y
conda activate intcort
pip install -e .
```

The code requires a CUDA-enabled GPU. Our experiments were conducted on NVIDIA A800 80GB GPUs.

## Model Preparation

Download or prepare the following two models, and pass their local paths to the scripts:

```text
Qwen2-VL-7B-Instruct
Qwen2.5-1.5B-Instruct
```

Qwen2-VL-7B performs visual inference, while Qwen2.5-1.5B-Instruct is used as the spatial query classifier.

## Quick Start

Run INTCORT on a single spatial reasoning sample:

```bash
python scripts/infer.py \
  --image path/to/image.jpg \
  --question "Where is the mug in relation to the knife?" \
  --candidate-relations left right front behind \
  --vlm-path /path/to/Qwen2-VL-7B-Instruct \
  --router-path /path/to/Qwen2.5-1.5B-Instruct
```

## Evaluation

Prepare a JSONL file in the following format:

```json
{"id":"sample-1","image":"images/example.jpg","question":"Where is the mug in relation to the knife?","answer":"left","candidate_relations":["left","right","front","behind"]}
```

Then run:

```bash
python scripts/evaluate.py \
  --manifest path/to/manifest.jsonl \
  --image-root path/to/dataset \
  --output outputs/predictions.json \
  --vlm-path /path/to/Qwen2-VL-7B-Instruct \
  --router-path /path/to/Qwen2.5-1.5B-Instruct
```

Datasets and model weights are not included in this repository.
