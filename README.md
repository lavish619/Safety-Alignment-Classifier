# Toxicity Classification with Adversarial Robustness

## 📌 Project Overview
This project focuses on training a **robust toxicity classification model** that can withstand **adversarial attacks**. The training pipeline includes:
- **Baseline Classifier:** Fine-tuned transformer-based model.
- **Adversarial Perturbations:** Applying adversarial attacks on text inputs.
- **Reinforcement Learning (RL) Fine-Tuning:** Using RL to improve robustness.

## ⚙️ Environment Setup

### 1️⃣ Install Required Dependencies
First, ensure you have **Python 3.8+** installed. Then manually install the required libraries:

```bash
pip install transformers datasets torch torchmetrics accelerate tqdm kaggle spacy
```

### 2️⃣ Download Pretrained SpaCy Models
The project relies on **SpaCy's word embeddings**. Run the following commands to download the required models:

```bash
python -m spacy download en_core_web_lg
python -m spacy download en_core_web_md
```

### 3️⃣ Set Up Kaggle API Key
To download the dataset from Kaggle, you need to configure your **Kaggle API key**.

1. Go to [Kaggle API Tokens](https://www.kaggle.com/account) and download `kaggle.json`.
2. Move it to the correct location:
   ```bash
   mkdir -p ~/.kaggle
   mv path/to/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

## 📂 Data Setup
Download the dataset using:

```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge --force
unzip -o jigsaw-toxic-comment-classification-challenge.zip -d ./jigsaw_toxicity_data
unzip -o ./jigsaw_toxicity_data/train.csv.zip -d ./jigsaw_toxicity_data
unzip -o ./jigsaw_toxicity_data/test.csv.zip -d ./jigsaw_toxicity_data
unzip -o ./jigsaw_toxicity_data/test_labels.csv.zip -d ./jigsaw_toxicity_data
```

## 🚀 Running the Project

### 1️⃣ Training the Baseline Classifier
```bash
python train.py --data_directory ./jigsaw_toxicity_data --model_path ./models/classifier.pt
```

### 2️⃣ Evaluation and Testing against Adversarial Examples
```bash
python evaluate.py --data_directory ./jigsaw_toxicity_data --model_path ./models/classifier.pt --adversarial
```

### 3️⃣ RL-Based Fine-Tuning for Robustness and Evaluation
```bash
python rl_policy.py --data_directory ./jigsaw_toxicity_data --classifier_model_path ./models/classifier.pt --policy_model_path ./models/policy.pt
```

## 🔧 Project Structure
```
📂 Safety-Alignment-Classifier
│── 📂 jigsaw_toxicity_data      # Contains dataset files
│── 📂 models                    # Saved models
│── 📂 src                       # Training & inference scripts
    |── dataset.py               # Dataset creation
    |── model.py                 # Model creation 
    │── train.py                 # Baseline classifier training
    │── evaluate.py              # Model evaluation
    │── rl_policy.py             # RL fine-tuning and evaluation for robustness
```

