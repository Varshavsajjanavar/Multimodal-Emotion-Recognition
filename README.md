
# Multimodal Emotion Recognition using Speech and Text

## Overview

This project implements a complete multimodal emotion recognition system that classifies human emotions using speech, text, and a combination of both modalities. The system is built on the Toronto emotional speech set and compares three model variants:

* Speech-only emotion recognition
* Text-only emotion recognition
* Multimodal fusion of speech and text

The goal is to study how different modalities contribute to emotion classification and how fusion improves robustness.

---

## Dataset

The Toronto Emotional Speech Set (TESS) contains acted emotional speech recordings labeled with seven emotions:

* angry
* disgust
* fear
* happy
* neutral
* sad
* surprise

Each audio file includes a corresponding transcript derived from the filename. The dataset is balanced across classes and is split using a stratified strategy:

* 72% training
* 8% validation
* 20% test

This ensures equal emotion distribution across splits.

---

## System Architecture

The project is organized into three modular pipelines:

### 1. Speech Pipeline

* Audio resampling to 16 kHz
* Silence trimming and fixed-length padding
* Mel-spectrogram feature extraction
* Bidirectional LSTM for temporal modeling
* Linear classifier for emotion prediction

### 2. Text Pipeline

* Tokenization using a pretrained transformer tokenizer
* Contextual embeddings from BERT base uncased
* Classification using the [CLS] representation

### 3. Fusion Pipeline

* Concatenation of speech and text embeddings
* Fully connected classifier for multimodal prediction

---

## Project Structure

```
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   ├── text_pipeline/
│   │   ├── train.py
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── train.py
│       └── test.py
│
├── Results/
│   ├── accuracy_tables/
│   └── plots/
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install dependencies:

```
git clone <your-repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

---

## Running the Pipelines

### Train Speech Model

```
python models/speech_pipeline/train.py
```

### Test Speech Model

```
python models/speech_pipeline/test.py
```

### Train Text Model

```
python models/text_pipeline/train.py
```

### Test Text Model

```
python models/text_pipeline/test.py
```

### Train Fusion Model

```
python models/fusion_pipeline/train.py
```

### Test Fusion Model

```
python models/fusion_pipeline/test.py
```

---

## Results Summary

| Model  | Accuracy | F1 Score |
| ------ | -------- | -------- |
| Speech | 0.9821   | 0.9821   |
| Text   | 1.0000   | 1.0000   |
| Fusion | 1.0000   | 1.0000   |

The text and fusion models achieve perfect classification on this dataset, while the speech model also performs strongly.

---

## Evaluation

The project includes:

* Accuracy and weighted F1 score
* Confusion matrices
* Error analysis
* t-SNE visualization of learned embeddings

Plots and tables are available in the `Results/` directory.

---

## Reproducibility

The repository provides:

* Modular training and testing scripts
* Fixed dataset splits
* Saved checkpoints
* Automated preprocessing
* Clear directory organization

Running the scripts sequentially reproduces all experiments and results.

---

## Future Work

Possible extensions include:

* Attention-based fusion strategies
* Larger speech encoders
* Evaluation on spontaneous conversational datasets
* Cross-speaker generalization experiments

---

Name : Varsha Sajjanavar
