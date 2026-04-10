# Multimodal Corruptions

Code for the work: **Evaluating the Robustness of Vision-Language Models Under Simultaneous Multimodal Perturbations**

## Overview

This project investigates how Vision-Language Models (VLMs) perform when both visual and textual inputs are corrupted simultaneously. While prior work evaluates robustness to unimodal perturbations in isolation, real-world deployments (e.g., autonomous driving) expose both modalities to degradation at once. We frame object detection as a Visual Question Answering (VQA) task and apply parameterized corruptions to probe whether VLMs compensate via cross-modal grounding or collapse under joint corruption.

## Key Contributions

- **Continuous-severity perturbation framework**: 12 visual corruptions (blur, noise, color, weather, occlusion, digital) and 8 textual corruptions (character, word, semantic level), each parameterized by a continuous severity `S in [0, 1]`.
- **Two NSGA-II multi-objective optimization variants**: a disjoint paired attack (Variant A) and a simultaneous budget-constrained attack (Variant B), optimizing detection degradation vs. perceptual stealth.
- **SWAD metric**: Stealth-Weighted Attack Degradation, a composite score balancing attack effectiveness with imperceptibility.
- **Stratified evaluation dataset**: 750 curated samples from ILSVRC 2017, split into isolated, clustered, and heterogeneous scenes.

## Repository Structure

```
methodology/    Perturbation framework and optimization setup
multimodal/     Multimodal (joint) perturbation experiments
unimodal/       Unimodal baseline experiments
selection/      Dataset curation and stratified sampling
results/        Analysis and evaluation notebooks
```
