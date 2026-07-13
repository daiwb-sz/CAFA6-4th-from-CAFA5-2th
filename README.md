# CAFA6 Project README
> **Code Source Note**: The backbone code in this repository is primarily derived from the second‑place solution of the CAFA5 Protein Function Prediction Competition. Original repository address:
> [https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place](https://github.com/btbpanda/CAFA5-protein-function-prediction-2nd-place)

## 📁 Project Overview & Data Instructions
This repository contains experimental codes and complete data pipelines for the CAFA6 task. All empty folders in the project are pre‑defined directories for data storage and can be used directly without manual creation or modification.

All raw source data, intermediate processed files, and experimental results of each step are fully released and available for download via the official link below:

**Download Link: https://huggingface.co/buckets/SSDASF/CAFA6**

The link covers full‑cycle project files, including original raw data, preprocessed intermediate data, and step‑by‑step experimental outputs, which can be directly adapted and applied to the project code for complete experimental reproduction.

## 💻 Running Environment Configuration
The software environment of this project is consistent with the official standard environment of the **CAFA5‑2th team**. No additional environment adaptation is required, ensuring stable compatibility.

### Hardware Requirements
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- This hardware configuration fully supports the full process of model training, validation, and inference, and meets the video memory requirements for large‑scale data processing and model computation.

### Software Requirements
This project inherits the general dependency environment of CAFA5‑2th from the original repo by btbpanda. You can build the running environment according to their official configuration. All core frameworks and dependency library versions are completely compatible with the project codes and data without version conflicts.

## 📌 Usage Guide
1. Download the full data files from the provided Hugging Face link;
2. Place the downloaded data and result files into the corresponding pre‑defined empty folders of the project;
3. Run the project code under the standard CAFA5‑2th environment to reproduce the complete experimental process and results.
