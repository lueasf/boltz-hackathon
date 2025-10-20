# Boltz Hackathon — Antibody–Antigen Complex Prediction

## 1. Introduction

This document presents the work completed for the **Boltz Hackathon**, organized by **EMD Group** and the **Massachusetts Institute of Technology (MIT)**.  
The competition focused on advancing **protein–protein** and **protein–ligand** complex prediction using the **Boltz** open-source framework, a successor to the AlphaFold family of models.

The task involved adapting and improving the Python entry point `hackathon/predict_hackathon.py` to optimize Boltz configuration strategies, sampling diversity, and post-processing pipelines for the **Antibody–Antigen complex prediction challenge**.

All experiments were executed on **AWS SageMaker**, using an **NVIDIA L4 GPU (24 GB VRAM)** within the provided `boltz` conda environment.


## 2. Background
### 2.1 From AlphaFold to Boltz

In **2020**, **DeepMind** introduced **AlphaFold2**, a model that revolutionized single-protein structure prediction.  
By **2024**, **AlphaFold3** extended this success to multi-chain and multi-molecule complexes (proteins, RNA, ligands). However, it remained closed-source.

In response, **Boltz** emerged as the first **publicly available AlphaFold3-like model**, designed for generalized molecular structure prediction using a **diffusion-based approach**.  
It retained the core attention-based architecture of AlphaFold but replaced the deterministic folding pipeline with stochastic **Boltz diffusion sampling**.

## 3. Boltz Architecture Overview

Boltz integrates three main modules:

1. **Trunk Network**  
   Handles sequence and pairwise encodings using MSA attention, template embedding, and geometric triangle updates.

2. **Confidence Module**  
   Outputs **pLDDT** and **ipTM** metrics, measuring the predicted local accuracy and inter-chain interface confidence.

3. **Affinity Module**  
   Estimates the probability that two molecules bind, predicting binding energies and likelihoods for re-ranking.

Boltz’s **diffusion process** incrementally refines noisy atom coordinates through learned potential fields, while a **recycling loop** enforces internal consistency across iterations.

## 4. Hackathon Context

### 4.1 Challenge Definition

The hackathon required participants to predict **Antibody–Antigen complexes**, a difficult task in computational biology due to the high flexibility of antibody binding regions.

Each sample included:
- Chain **H** (heavy antibody chain),
- Chain **L** (light antibody chain),
- Chain **A** (antigen),
- Precomputed **MSAs** for each chain,
- Ground-truth PDB structures for evaluation.

### 4.2 Evaluation Metrics

Predictions were scored with **CAPRI-Q**, classifying each model as:
- **High**, **Medium**, **Acceptable**, or **Incorrect**  
based on structural alignment (RMSD) and interface similarity.  
The goal was to maximize the number of top-ranked predictions classified as *Acceptable or better* (I failed).

## 5. Methodology

### 5.1 Python Implementation

Two key functions in `predict_hackathon.py` were reimplemented.

#### `prepare_protein_complex()`

This function creates multiple Boltz configurations by modifying CLI arguments dynamically.  
Implemented strategies:

- MSA subsampling control (`--num_subsampled_msa` = 256 or full MSA)
- Diverse diffusion sampling (`--diffusion_samples` = 6–8)
- Extended refinement (`--recycling_steps` = 6–8)
- Adjusted diversity scaling (`--step_scale` = 0.6–0.8)
- Randomized seeds for reproducibility

```python
config_params = [
  {"subsample_msa": False, "num_msa": None, "step_scale": 0.6, "diffusion_samples": 8, "recycling_steps": 8},
  {"subsample_msa": True,  "num_msa": 256,  "step_scale": 0.6, "diffusion_samples": 8, "recycling_steps": 6},
]
```

same `for post_process_protein_complex()`.

### 5.2 Environment
- Platform: AWS SageMaker Studio
- GPU: NVIDIA L4 (24 GB VRAM)
- Driver: CUDA 13.0 / Driver 580.95.05
- Precision: bfloat16 with Automatic Mixed Precision (AMP)
- Python Environment: Conda (boltz), Python 3.11

Evaluation with : 
```bash
python hackathon/evaluate_abag.py \
  --dataset-file /tmp/one_abag.jsonl \
  --submission-folder _pred_smoke \
  --result-folder _res_smoke
```

## 6. Results
The final evaluation yielded “Incorrect” classification on the validation datapoint (8CYH).
While the predicted structure generation succeeded, the geometric alignment to the ground truth was insufficient for a CAPRI-Q “acceptable” threshold.

However, this outcome confirms that:

- The pipeline executes end-to-end successfully;
- The MSA processing, configuration management, and Boltz CLI interfacing are correct;
- The system is ready for large-scale testing on the official validation dataset.


## Glossary
| Term                                  | Definition                                                                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **AlphaFold2/3**                      | DeepMind’s neural networks for protein structure prediction. AF2 covers single chains; AF3 extends to complexes.                  |
| **Boltz**                             | An open-source model inspired by AlphaFold3, implementing diffusion-based structural generation with binding affinity prediction. |
| **Diffusion Sampling**                | A generative process where structures are iteratively denoised from random noise under physical and learned constraints.          |
| **Recycling Steps**                   | Iterative refinement passes through the network to improve structure consistency.                                                 |
| **MSA (Multiple Sequence Alignment)** | Alignments of homologous sequences providing evolutionary constraints to improve prediction accuracy.                             |
| **pLDDT**                             | Per-residue confidence score predicting the reliability of local structure.                                                       |
| **ipTM**                              | Interface predicted TM-score estimating the accuracy of inter-chain contacts.                                                     |
| **CAPRI-Q**                           | Docking evaluation metric classifying predictions as high, medium, acceptable, or incorrect.                                      |
| **AWS SageMaker**                     | Managed machine-learning platform used to train and run the model in cloud environments.                                          |
| **NVIDIA L4 GPU**                     | Data center GPU optimized for AI inference and model experimentation.                                                             |


## Conclusion 

Through this project, the full Boltz prediction pipeline was successfully deployed, customized, and executed on a cloud-based GPU infrastructure.
Despite the “incorrect” classification, the work demonstrates a functional and reproducible end-to-end Boltz inference and evaluation setup — a necessary foundation for further experimentation in antibody–antigen complex modeling.

The hackathon provided practical experience in integrating deep learning for structural biology, exploring bio-ML workflows, and understanding the bridge between computational protein design and biophysical validation.
