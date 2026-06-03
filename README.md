# Hybrid Neuro-Symbolic Arabic Simile Detection
## Overview

This project implements a hybrid system for detecting similes in Arabic text. It combines:

- A **symbolic rule-based detector** for explicit linguistic patterns
- A **transformer-based model (AraBERTv2)** for contextual representation
- A **neural attention mining module** to extract interpretable lexical signals
- A **fusion model** that combines symbolic and neural outputs

The goal is to improve both **classification performance and interpretability** in Arabic rhetorical analysis.

---

## Task Definition

Given an Arabic sentence, the system performs:

1. **Binary classification**
   - Detect whether the sentence contains a simile or not

2. **Structure extraction (symbolic layer)**
   - Identify:
     - Subject (المشبّه)
     - Particle (أداة التشبيه)
     - Object (المشبّه به)

3. **Explainability**
   - Provide symbolic rules and neural attention evidence supporting the prediction

---
## Dataset

The dataset used in this project is an Arabic simile dataset designed for binary classification (simile vs non-simile).

The final data split is:

- Training set: 6,000 samples  
- Validation set: 1,108 samples  
- Test set: 3,000 samples  

The dataset contains a mixture of:

- Short social media-style sentences  
- Medium-length natural language sentences  
- Linguistically diverse Arabic expressions involving simile structures  

This variability requires robust preprocessing and contextual modeling to handle differences in sentence length and structure.
---

## Preprocessing

Arabic text is normalized before training using the following steps:

- Remove diacritics
- Normalize Alef variants (آ, أ, إ → ا)
- Normalize Teh Marbuta (ة → ه)
- Normalize Alef Maksura (ى → ي)
- Remove punctuation
- Collapse repeated whitespace
- Trim text

This follows standard Arabic NLP preprocessing pipelines used in transformer-based systems.

---

## Model Architecture

### 1. Symbolic Module

A rule-based system that detects simile structures using:

- Strong particles: كأن، كأنما
- Weak particles: كما، مثل، كـ
- Verbs: يشبه، يماثل، يحاكي
- Nouns: شبيه، مثيل، نظير
- Prefix patterns: كـ + noun

Outputs:
- Extracted structures
- Rule-based confidence scores
- Probabilistic fusion over multiple rules

---

### 2. Neural Module (AraBERTv2)

- Base model: `aubmindlab/bert-large-arabertv02`
- 110M parameter transformer
- Fine-tuned for binary classification

Outputs:
- Sentence-level classification probability
- CLS embedding representation
- Attention weights (used for analysis)

---

### 3. Neural Attention Feature Mining

Attention maps are used to derive interpretable features:

- **Attention to discriminative tokens**
  - Measures focus on tokens statistically associated with similes

- **Attention entropy**
  - Measures uncertainty / dispersion of attention

- **Positional bias**
  - Measures whether attention is focused on early or late tokens

Additionally, high-attention tokens are used to derive **likelihood-based lexical rules**.

---

### 4. Symbolic Feature Representation

Each sentence is converted into an 11-dimensional vector:

- Presence of simile structure
- Number of detected structures
- Maximum rule confidence
- Average rule confidence
- Rule-type indicators (particle, verb, noun, prefix)
- Average distance between subject and object
- Multi-simile flag

---

### 5. Hybrid Fusion Model

Final prediction is computed as:

Final Score =
α × (AraBERT probability)
(1 − α) × (Symbolic score)


Where:
- α = 0.7 (default)
- Symbolic score comes from rule-based detector

---

## Training Strategy

### Phase 1: Frozen Transformer
- AraBERT frozen
- Train classification head only
- Learning rate: 1e-4

### Phase 2: Fine-tuning
- Unfreeze AraBERT
- Fine-tune full model
- Learning rate: 2e-6

Early stopping is applied based on validation loss.

---

## Evaluation

### Symbolic baseline
- Accuracy: ~0.76
- Strong interpretability
- Limited generalization

### Hybrid model
- Validation accuracy: ~0.94
- Test accuracy: ~0.76
- Improved robustness over symbolic-only system

---

## Explainability

Each prediction includes:

### 1. Symbolic explanation
- Detected rule(s)
- Extracted simile components
- Rule confidence

### 2. Neural explanation
- Top-attended tokens
- Attention entropy
- Positional bias

### 3. Hybrid decision
- AraBERT probability
- Symbolic probability
- Final fused score

---

## Key Components

- `symbolic_detector(sentence)` → rule-based detection
- `build_symbolic_matrix(texts)` → symbolic feature extraction
- `extract_attention_patterns()` → CLS attention extraction
- `mine_neural_rules()` → attention-based lexical mining
- `HybridAraBERT` → fusion classifier
- `explain_prediction(sentence)` → full explainability pipeline

---

## Saved Artifacts

- `saved_model/` → trained hybrid model
- `discriminative_tokens.json` → learned neural lexical rules
- `*_cls_attn.npy` → cached attention matrices

---

## Related Work
This system is aligned with Arabic NLP shared-task research on:
- Arabic rhetorical device detection
- Transformer-based classification models (AraBERT, MARBERT)
- Neuro-symbolic reasoning systems
- Explainable NLP for low-resource languages

---
## Baseline Model

For comparison, we used an AraBERT-based baseline model developed separately:

- AraBERT Baseline: https://github.com/arwghm/Al-Bayan-Baseline-Model

The baseline uses AraBERT for Arabic simile classification without the neuro-symbolic reasoning layer implemented in AlBayan.
## Requirements

```bash
pip install tensorflow transformers arabert-preprocess
pip install scikit-learn pandas numpy matplotlib pyarabic 
