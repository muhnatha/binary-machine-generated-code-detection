# Cross-Modal Representation Learning with UniXcoder and AST Structural Features for Machine-Generated Code Detection

**Binary Code Classification of Human-Written or Machine-Generated Code**

This repository contains the implementation for the research project developed for **SemEval-2026 Task 13 SubTask A**. The project distinguishes between human-written and machine-generated code using a deep learning approach that combines the semantic capabilities of **UniXcoder** with the structural insights of **Abstract Syntax Trees (AST)**.

## 📌 Project Overview
The rapid advancement of Large Language Models (LLMs) has led to code generation that closely mimics human code, raising concerns about software security and integrity. This project develops a binary detection system to classify code as either **Human-Written** or **Machine-Generated**.

**Methodology:**
We utilize a hybrid architecture:
* **UniXcoder**: Used for semantic understanding of the source code.
* **AST Integration**: Structural features are extracted using `tree-sitter` and flattened into sequences to capture syntactic patterns often missed by text-only models.

**Shared Task:** [SemEval-2026 Task 13](https://github.com/mbzuai-nlp/SemEval-2026-Task13)

## 📊 Dataset
The project uses the **SemEval-2026-Task13-TaskA** dataset:
* **Training**: 5,000 samples (Balanced: ~48% Human / ~52% Machine)
* **Validation**: 2,000 samples
* **Test**: 1,000 samples
* **Languages**: Predominantly Python (~90%), with minority languages including C++, Java, and Go.

## 📈 Experimental Results
Our experiments compared a **Code-only** approach against our proposed **Code + AST** method. The addition of AST features significantly improved generalization and stability.

| Method | Precision | Recall | Accuracy | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| UniXcoder (Code-only) | 0.57 | 0.57 | 0.39 | 0.39 |
| **UniXcoder (Code + AST)** | **0.55** | **0.57** | **0.49** | **0.48** |

**Key Findings:**
1. **Improved Generalization**: The `Code + AST` model increased Accuracy from 0.39 to 0.49.
2. **Language Sensitivity**: The method proved highly effective for strictly structured languages (C++, C#, Java) but showed less improvement for dynamic languages like Python and Go.
3. **Stability**: The AST-enhanced model demonstrated better stability against overfitting compared to the code-only baseline.

## 🛠️ Requirements & Setup

### Requirements
1. Python `3.10.x`
2. Editor compatible with Python and Jupyter Notebook (e.g., VSCode, PyCharm)
3. CUDA-enabled GPU (Recommended for training)

### How to Setup the Environment

1. **Clone the repository**
   ```bash
   git clone [https://github.com/muhnatha/binary-machine-generated-code-detection.git](https://github.com/muhnatha/binary-machine-generated-code-detection.git)
   cd binary-machine-generated-code-detection
