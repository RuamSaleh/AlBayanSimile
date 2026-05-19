"""
Arabic Simile Detector — FastAPI
POST /predict  →  prediction + full symbolic + neural explanation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
from arabert.preprocess import ArabertPreprocessor
from collections import defaultdict
import json, os, logging

#  SYMBOLIC MODULE  (from symbolicModel.py)
from symbolicModel import (
    build_symbolic_matrix,
    symbolic_detector,
    RULE_CONFIDENCE,
)

logging.getLogger("tensorflow").setLevel(logging.ERROR)

#  CONFIG  — edit paths to match your deployment
MODEL_NAME        = "aubmindlab/bert-large-arabertv02"
SAVED_MODEL_PATH  = "saved_model/"
DISC_TOKENS_PATH  = "discriminative_tokens.json"
MAX_LEN           = 32

app = FastAPI(
    title="Arabic Simile Detector",
    description="Hybrid AraBERT + symbolic model for Arabic simile detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#  STARTUP — load heavy objects once
arabert_prep        = None
tokenizer_obj       = None
arabert_attention   = None
hybrid_model        = None
infer               = None          # ← saved model signature
discriminative_tokens: dict = {}


@app.on_event("startup")
def load_models():
    global arabert_prep, tokenizer_obj, arabert_attention, \
           hybrid_model, infer, discriminative_tokens

    print("Loading tokenizer + preprocessor …")
    arabert_prep  = ArabertPreprocessor(model_name=MODEL_NAME)
    tokenizer_obj = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading AraBERT for attention extraction …")
    arabert_attention = TFAutoModel.from_pretrained(
        MODEL_NAME, output_attentions=True, from_pt=True
    )

    print("Loading hybrid model …")
    hybrid_model = tf.saved_model.load(SAVED_MODEL_PATH)
    infer        = hybrid_model.signatures["serving_default"]

    if os.path.exists(DISC_TOKENS_PATH):
        with open(DISC_TOKENS_PATH) as f:
            discriminative_tokens = json.load(f)
        print(f"Loaded {len(discriminative_tokens)} discriminative tokens.")
    else:
        print("WARNING: discriminative_tokens.json not found — neural attention "
              "features will be zero.")

    print("All models ready.")


#  HELPERS

def encode(sentence: str):
    enc = tokenizer_obj(
        [arabert_prep.preprocess(sentence)],
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="tf",
    )
    return enc["input_ids"], enc["attention_mask"]


def get_cls_attention(input_ids, attention_mask) -> np.ndarray:
    outputs = arabert_attention(
        input_ids=input_ids,
        attention_mask=attention_mask,
        training=False,
        return_dict=True,
    )
    attn_stack = tf.stack(outputs.attentions, axis=0)
    cls_attn   = attn_stack[:, :, :, 0, :]
    cls_mean   = tf.reduce_mean(cls_attn, axis=[0, 2])
    return cls_mean.numpy()


def build_neural_feats(input_ids, cls_attn: np.ndarray) -> np.ndarray:
    disc_ids = set(
        tokenizer_obj.convert_tokens_to_ids(list(discriminative_tokens.keys()))
    )
    features = []
    for idx in range(cls_attn.shape[0]):
        attn_row = cls_attn[idx]

        attn_to_key = 0.0
        for pos in range(attn_row.shape[0]):
            if int(input_ids[idx, pos].numpy()) in disc_ids:
                attn_to_key = max(attn_to_key, float(attn_row[pos]))

        norm    = attn_row / (attn_row.sum() + 1e-10)
        entropy = float(-np.sum(norm * np.log(norm + 1e-10)))

        mid      = attn_row.shape[0] // 2
        pos_bias = float(attn_row[:mid].mean() - attn_row[mid:].mean())

        features.append([attn_to_key, entropy, pos_bias])

    return np.array(features, dtype=np.float32)


def top_attended_tokens(input_ids, attn_row: np.ndarray, k: int = 5):
    top_pos = np.argsort(attn_row)[-k:][::-1]
    result  = []
    for pos in top_pos:
        token = tokenizer_obj.decode([int(input_ids[0, pos].numpy())]).strip()
        if token in ["[CLS]", "[SEP]", "[PAD]", ""]:
            continue
        result.append({
            "token":       token,
            "attention":   round(float(attn_row[pos]), 4),
            "neural_rule": token in discriminative_tokens,
        })
    return result


#  SCHEMAS

class PredictRequest(BaseModel):
    sentence: str

    class Config:
        json_schema_extra = {
            "example": {"sentence": "الرجل مثل الأسد في الشجاعة"}
        }


class SymbolicStructure(BaseModel):
    rule: str
    subject: str | None
    particle: str
    object: str
    confidence: float


class SymbolicLayer(BaseModel):
    has_structure:       bool
    num_structures:      int
    max_confidence:      float
    avg_confidence:      float
    has_strong_particle: bool
    has_weak_particle:   bool
    has_verb:            bool
    has_noun:            bool
    has_prefix:          bool
    avg_distance:        float
    multi_simile_flag:   bool
    structures:          list[SymbolicStructure]


class AttendedToken(BaseModel):
    token:       str
    attention:   float
    neural_rule: bool


class NeuralLayer(BaseModel):
    attn_to_key_tokens: float
    attention_entropy:  float
    attention_pos_bias: float
    top_tokens:         list[AttendedToken]
    interpretation:     dict[str, str]


class PredictResponse(BaseModel):
    sentence:       str
    prediction:     str
    is_simile:      bool
    arabert_prob:   float
    hybrid_score:   float
    symbolic_layer: SymbolicLayer
    neural_layer:   NeuralLayer


#  ENDPOINT

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.sentence.strip():
        raise HTTPException(status_code=400, detail="Sentence must not be empty.")

    sentence = req.sentence.strip()

    # ── 1. Symbolic layer ──────────────────────
    sym_result = symbolic_detector(sentence)
    sym_feat   = build_symbolic_matrix([sentence])[0]   # (11,)

    structures = [
        SymbolicStructure(
            rule=s.rule,
            subject=s.subject,
            particle=s.particle,
            object=s.object,
            confidence=round(s.confidence, 3),
        )
        for s in sym_result.structures
    ]

    symbolic_layer = SymbolicLayer(
        has_structure=       bool(sym_feat[0]),
        num_structures=      int(sym_feat[1]),
        max_confidence=      round(float(sym_feat[2]), 3),
        avg_confidence=      round(float(sym_feat[3]), 3),
        has_strong_particle= bool(sym_feat[4]),
        has_weak_particle=   bool(sym_feat[5]),
        has_verb=            bool(sym_feat[6]),
        has_noun=            bool(sym_feat[7]),
        has_prefix=          bool(sym_feat[8]),
        avg_distance=        round(float(sym_feat[9]), 3),
        multi_simile_flag=   bool(sym_feat[10]),
        structures=          structures,
    )

    # ── 2. Neural layer ────────────────────────
    ids, mask   = encode(sentence)
    cls_attn    = get_cls_attention(ids, mask)
    neural_feat = build_neural_feats(ids, cls_attn)

    attn_to_key = float(neural_feat[0][0])
    entropy     = float(neural_feat[0][1])
    pos_bias    = float(neural_feat[0][2])

    interp = {
        "discriminative_focus": (
            "Focused on simile indicators (reliable)"
            if attn_to_key > 0.1
            else "Did not focus on known simile indicators"
        ),
        "confidence": (
            "Attention focused — confident"
            if entropy < 4.0
            else ("Attention moderate — uncertain" if entropy < 4.5
                  else "Attention scattered — very uncertain")
        ),
        "position_bias": (
            "Early-token focus (normal for Arabic similes)"
            if pos_bias > 0.01
            else "Late-token focus (unusual)"
        ),
    }

    neural_layer = NeuralLayer(
        attn_to_key_tokens= round(attn_to_key, 4),
        attention_entropy=  round(entropy, 4),
        attention_pos_bias= round(pos_bias, 4),
        top_tokens=         top_attended_tokens(ids, cls_attn[0]),
        interpretation=     interp,
    )

    # ── 3. Hybrid model ────────────────────────
    combined = np.hstack([
        sym_feat.reshape(1, -1),
        neural_feat,
    ]).astype(np.float32)

    result = infer(
        input_ids=          tf.cast(ids, tf.int32),
        attention_mask=     tf.cast(mask, tf.int32),
        combined_features=  tf.constant(combined, dtype=tf.float32),
    )

    # Print keys on first run if you need to verify: print(list(result.keys()))
    final_prob = float(result["output_0"][0][0])
    bert_prob  = float(result["bert_probability"][0][0])
    is_simile  = final_prob > 0.5

    return PredictResponse(
        sentence=       sentence,
        prediction=     "Simile" if is_simile else "Not Simile",
        is_simile=      is_simile,
        arabert_prob=   round(bert_prob,  3),
        hybrid_score=   round(final_prob, 3),
        symbolic_layer= symbolic_layer,
        neural_layer=   neural_layer,
    )


#  HEALTH CHECK

@app.get("/health")
def health():
    return {"status": "ok"}