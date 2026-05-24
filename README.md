# AlBayanSimile
# Hybrid Arabic Simile Detection using Symbolic Rules + AraBERT Attention Mining

This project introduces a **hybrid AI system for Arabic simile detection** that combines:

- 🧠 Symbolic linguistic rules (interpretable grammar detection)
- 🤖 AraBERT transformer representations (deep contextual learning)
- 🔍 Attention-based neural rule mining (discovering linguistic patterns from the model itself)

The goal is to build a system that is not only accurate, but also **explainable and linguistically grounded**.

---

## 🚀 Key Idea

Instead of relying only on a neural model or only handcrafted rules, this system merges both:

> Symbolic reasoning = interpretability  
> Neural learning = generalization  

The result is a **hybrid interpretable NLP classifier** for Arabic similes.

---

## 🧠 System Architecture

The pipeline consists of four main layers:

### 1. Symbolic Layer (Rule-Based Engine)
Detects similes using handcrafted Arabic linguistic rules:

- Explicit particles: (كأن، كأنما)
- Weak particles: (كما، مثل، كـ)
- Verbs: (يشبه، يماثل، يحاكي)
- Nominal patterns: (شبيه، مثيل)

Outputs:
- Structured simile components (subject / particle / object)
- Rule confidence scores

---

### 2. Neural Layer (AraBERT)
Uses **aubmindlab/bert-large-arabertv02**:

- Extracts contextual embeddings
- Produces classification probability
- Provides attention maps for interpretability

---

### 3. Attention Mining Layer
This is the core research contribution.

We analyze AraBERT attention weights to:

- Identify tokens strongly associated with similes
- Compute likelihood ratios between simile vs non-simile contexts
- Extract **discriminative linguistic signals automatically**

Outputs:
- Neural rule set (data-driven lexical indicators)

---

### 4. Hybrid Fusion Layer
Final prediction is computed as:

- Weighted combination of:
  - Neural probability (AraBERT head)
  - Symbolic score (rule-based engine)

This creates a balance between:
- Generalization (BERT)
- Interpretability (rules)

---

## 📊 Dataset

- Arabic simile dataset
- Train: 6000 samples
- Validation: 1108 samples
- Test: 3000 samples

---

## ⚙️ Feature Engineering

Each sentence is represented using:

### Symbolic Features (11D)
- Presence of simile structures
- Rule types activated
- Confidence scores
- Token distance metrics
- Multi-simile detection

### Neural Attention Features (3D)
- Attention to discriminative tokens
- Attention entropy (uncertainty)
- Positional bias (early vs late focus)

Final vector: **14-dimensional hybrid representation**

---

## 📈 Results

### Symbolic Model Only
- Accuracy: ~0.76
- Strong interpretability but limited coverage

### Full Hybrid Model
- Validation Accuracy: **~0.94**
- Test Accuracy: **~0.76 (real-world generalization gap)**
- Significant improvement in structured understanding

---

## 🔍 Explainability

The system provides full interpretability:

Example:
Sentence: الولد يشبه الشمس في إشراقته
[Symbolic Layer]
Rule detected: verb_simile
Confidence: 0.927
[Neural Layer]
Attention focuses on: "يشبه", "في"
[Final Decision]
Simile: YES (0.87)

## 💡 Key Contributions
This project contributes:
1. Hybrid NLP Architecture
Combines symbolic rules with transformer learning.
2. Attention-Based Rule Mining
Extracts linguistic patterns directly from model attention.
3. Explainable Arabic NLP System
Every prediction is interpretable at:
grammar level
neural level
hybrid decision level
## 📌 Why this matters
Most NLP systems are:
Either accurate but black-box (BERT)
Or interpretable but weak (rules)
This system bridges both worlds.

## 👩‍💻 Author
Built by a Computer Science researchers
