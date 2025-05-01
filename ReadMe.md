# Towards a Deeper Understanding of Neural Distinguishers

This repository provides the supporting code for the paper "Towards a Deeper Understanding of Neural Distinguishers". It contains the implementations and training pipelines for neural distinguishers on lightweight block ciphers using two different model architectures and training strategies by Chen et al and Zhang et al. It embeds codes for explainable AI (XAI) analysis using LIME and SHAP.

## Directory Overview

Under the main folder, two approaches are provided:

### 1. Chen
- **Model and Training**:  
  - `deep_nt_mc.py` — Defines the model and training pipeline based on [Chen's method](https://doi.org/10.1093/comjnl/bxac019).
- **Data Generation**:  
  - `speck_nd_mc.py` — Generates training data for Chen's approach
- **Explainability (XAI)**:  
  - `sharp_explainer.py` — Generates SHAP explainability results.
  - `lime_explainer.py` — Generates LIME explainability results.
- **Execution**:  
  - `train_distingusher.py` — Main script to train the Chen's model directly.

### 2. Zhang
- **Model and Training**:  
  - `multiple_parallel_convolutional_layers_net.py` — Defines and trains the Inception-style model based on [Zhang's method](https://doi.org/10.62056/ay11wa3y6).
- **Data Generation**:  
  - `speck_inception.py` — Generates training data for Zhang's approach
- **Explainability (XAI)**:  
  - `sharp_explainer.py` and `lime_explainer.py` — Same tools as used for Chen.
