import streamlit as st
import tensorflow as tf
import numpy as np

from transformers import AutoTokenizer
from arabert.preprocess import ArabertPreprocessor

from symbolic_model import (
    build_symbolic_matrix,
    symbolic_detector
)

# ======================
# إعدادات
# ======================

MODEL_NAME = "aubmindlab/bert-large-arabertv02"
MAX_LEN = 120

# ======================
# تحميل المودل
# ======================

@st.cache_resource
def load_all():

    model = tf.keras.models.load_model(
    "/Users/roamsaleh/Desktop/AlBayanSimile/Model/arabert_simile_model",
    compile=False
)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    arabert_prep = ArabertPreprocessor(
        model_name=MODEL_NAME
    )

    return model, tokenizer, arabert_prep


model, tokenizer, arabert_prep = load_all()

# ======================
# preprocess
# ======================

def preprocess_text(text):

    text = arabert_prep.preprocess(text)

    encodings = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # 🔥 احذف token_type_ids لو موجود
    return input_ids, attention_mask
# ======================
# prediction
# ======================

def predict_simile(text):

    input_ids, attention_mask = preprocess_text(text)

    symbolic_features = build_symbolic_matrix([text])

    prediction = model(
    [input_ids, attention_mask, symbolic_features],
    training=False
)

    prob = float(prediction[0][0])

    label = "تشبيه" if prob >= 0.5 else "ليس تشبيه"

    # استخراج عناصر التشبيه
    evidence = symbolic_detector(text)

    structures = []

    for s in evidence.structures:

        structures.append({
            "المشبه": s.subject if s.subject else "غير محدد",
            "أداة التشبيه": s.particle,
            "المشبه به": s.object,
            "الثقة": round(float(s.confidence), 2),
            "القاعدة": s.rule
        })

    return label, prob, structures


# ======================
# واجهة المستخدم
# ======================

st.set_page_config(
    page_title="كاشف التشبيه العربي",
    layout="centered"
)

st.title("كاشف التشبيه العربي")

st.write("أدخل نصًا عربيًا لتحليل التشبيه واستخراج عناصره")

text = st.text_area(
    "النص العربي",
    height=180,
    placeholder="مثال: وجهه كالقمر في الجمال"
)

if st.button("تحليل النص"):

    if not text.strip():

        st.warning("الرجاء إدخال نص")

    else:

        with st.spinner("جاري التحليل..."):

            label, prob, structures = predict_simile(text)

        # ======================
        # النتيجة العامة
        # ======================

        st.subheader("النتيجة")

        if label == "تشبيه":
            st.success(f"تم اكتشاف تشبيه بنسبة {prob:.2%}")
        else:
            st.error(f"لا يوجد تشبيه واضح بنسبة {1 - prob:.2%}")

        # ======================
        # عناصر التشبيه
        # ======================

        if structures:

            st.subheader("عناصر التشبيه المكتشفة")

            for idx, s in enumerate(structures, start=1):

                with st.expander(f"تشبيه رقم {idx}"):

                    st.write(f"المشبه: {s['المشبه']}")
                    st.write(f"أداة التشبيه: {s['أداة التشبيه']}")
                    st.write(f"المشبه به: {s['المشبه به']}")
                    st.write(f"الثقة: {s['الثقة']}")
                    st.write(f"القاعدة: {s['القاعدة']}")

        else:

            st.info("لم يتم استخراج عناصر تشبيه واضحة")