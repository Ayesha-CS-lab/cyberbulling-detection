"""
Interactive web demo for the two-stage cyberbullying detection system.

Stage 1 (m-BERT / MuRIL) classifies each message as aggressive or not.
Stage 2 aggregates a conversation's aggression, repetition, intent and peerness
into the final cyberbullying decision (dense MLP).

Run:  python -m streamlit run demo.py
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer
from src.models.aggression_model import AggressionClassifier
from src.train_stage2 import CBClassifier, build_features
from src.data.preprocessing import TextPreprocessor
from src.utils.scoring import IntentScorer

# ── Config ──────────────────────────────────────────────────────────────────
MODELS = os.path.join(PROJECT_ROOT, 'models')
MBERT = 'bert-base-multilingual-cased'
MURIL = 'google/muril-base-cased'
AGG_PATHS = {'m-BERT': (MBERT, os.path.join(MODELS, 'aggression_mbert.pth')),
             'MuRIL':  (MURIL, os.path.join(MODELS, 'aggression_muril.pth'))}
CB_PATH = os.path.join(MODELS, 'cb_classifier.pth')
IMG_FUSION_PATH = os.path.join(MODELS, 'image_fusion_mbert.pth')
MAX_LEN = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRE = TextPreprocessor()
INTENT = IntentScorer()


# ── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner='Loading aggression model…')
def load_aggression(model_name, path):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AggressionClassifier(text_model_name=model_name).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return tok, model


@st.cache_resource(show_spinner='Loading image model…')
def load_image_model():
    if not os.path.exists(IMG_FUSION_PATH):
        return None
    from src.models.fusion_model import CyberbullyingDetector
    from torchvision import transforms
    itok = AutoTokenizer.from_pretrained(MBERT)
    model = CyberbullyingDetector(text_model_name=MBERT, num_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(IMG_FUSION_PATH, map_location=DEVICE))
    model.eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return itok, model, tf


@st.cache_resource(show_spinner='Loading cyberbullying model…')
def load_cb():
    if not os.path.exists(CB_PATH):
        return None
    ckpt = torch.load(CB_PATH, map_location=DEVICE, weights_only=False)
    model = CBClassifier(input_dim=ckpt['input_dim']).to(DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt['mean'], ckpt['std']


def aggression_prob(text, lang, tok, model):
    clean = PRE.preprocess(text, lang=lang)
    enc = tok(clean, add_special_tokens=True, max_length=MAX_LEN,
              padding='max_length', truncation=True, return_token_type_ids=True,
              return_tensors='pt')
    with torch.no_grad():
        logit = model(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE),
                      enc['token_type_ids'].to(DEVICE))
        return float(torch.sigmoid(logit).cpu().item())


def cb_predict(feat_row, cb):
    model, mean, std = cb
    X, _ = build_features(pd.DataFrame([feat_row]))
    Xs = (X - mean) / std
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(Xs, dtype=torch.float).to(DEVICE)))
    return float(p.cpu().item())


# ── UI ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='Cyberbullying Detection', page_icon='🛡️', layout='centered')
st.title('🛡️ Multilingual Cyberbullying Detection')
st.caption('Two-stage pipeline — Stage 1: message aggression (m-BERT / MuRIL) · '
           'Stage 2: relationship-level cyberbullying decision')

with st.sidebar:
    st.header('⚙️ Settings')
    available = [k for k, (_, p) in AGG_PATHS.items() if os.path.exists(p)]
    if not available:
        st.error('No aggression model found in models/. Add aggression_mbert.pth or aggression_muril.pth.')
        st.stop()
    choice = st.selectbox('Text encoder', available, index=len(available) - 1)
    lang = st.selectbox('Language', ['roman_urdu', 'en'], index=0)
    thr = st.slider('Aggression threshold', 0.1, 0.9, 0.5, 0.05)
    st.divider()
    st.markdown(f"**Device:** {DEVICE}")
    st.markdown(f"{'✅' if os.path.exists(CB_PATH) else '❌'} Stage 2 model")

model_name, path = AGG_PATHS[choice]
tok, agg_model = load_aggression(model_name, path)
cb = load_cb()

tab1, tab2, tab3 = st.tabs(['🔹 Single message (Stage 1)', '🔸 Conversation (full pipeline)',
                            '🖼️ Meme (image + text)'])

# ── Tab 1: single message ───────────────────────────────────────────────────
with tab1:
    st.subheader('Analyse one message')
    txt = st.text_area('Message', placeholder='e.g. tum bohut bure insaan ho', height=100,
                       key='single')
    if st.button('Analyse', type='primary') and txt.strip():
        prob = aggression_prob(txt, lang, tok, agg_model)
        intent = INTENT.get_detailed_analysis(txt)
        c1, c2 = st.columns(2)
        c1.metric('Aggression probability', f'{prob*100:.1f}%')
        c2.metric('Intent-to-harm score', f"{intent['score']*100:.1f}%")
        st.progress(prob)
        if prob >= thr:
            st.error('⚠️ Aggressive message')
        else:
            st.success('✅ Not aggressive')
        if intent['detected_categories']:
            st.caption('Intent cues: ' + ', '.join(intent['detected_categories'])
                       + '  ·  keywords: ' + ', '.join(intent['matched_keywords']))

# ── Tab 2: conversation → cyberbullying ─────────────────────────────────────
with tab2:
    st.subheader('Analyse a conversation (Sender → Target)')
    st.caption('One message per line. Stage 1 scores each message; Stage 2 combines '
               'aggression %, repetition, intent and peerness into a cyberbullying decision.')
    convo = st.text_area('Messages (one per line)', height=150, key='convo',
                         placeholder='tum bure ho\ntujhe dekh lunga\nbakwaas band kar')
    col = st.columns(3)
    peerness = col[0].slider('Peerness', 0.0, 1.0, 0.5, 0.05)
    u1_age = col[1].number_input('Sender age', 8, 60, 14)
    u2_age = col[2].number_input('Target age', 8, 60, 14)

    if st.button('Run full pipeline', type='primary') and convo.strip():
        if cb is None:
            st.error('Stage 2 model (cb_classifier.pth) not found.')
        else:
            msgs = [m for m in convo.splitlines() if m.strip()]
            rows, aggressive = [], 0
            max_intent = 0.0
            for m in msgs:
                p = aggression_prob(m, lang, tok, agg_model)
                is_agg = p >= thr
                aggressive += int(is_agg)
                max_intent = max(max_intent, INTENT.compute_score(m))
                rows.append({'message': m, 'aggression': f'{p*100:.0f}%',
                             'aggressive': '🔴' if is_agg else '🟢'})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            total = len(msgs)
            feat = {
                'total_messages': total, 'aggressive_count': aggressive,
                'aggression_ratio': aggressive / total if total else 0,
                'repetition_count': aggressive,
                'repetition_flag': 1 if aggressive >= 2 else 0,
                'aggression_active_days': aggressive,   # demo proxy (no timestamps)
                'aggression_span_days': aggressive,     # demo proxy
                'intent_to_harm': max_intent, 'peerness': peerness,
                'u1_age': u1_age, 'u1_grade': max(1, u1_age - 5),
                'u2_age': u2_age, 'u2_grade': max(1, u2_age - 5),
                'u1_gender': 'Male', 'u2_gender': 'Male',
            }
            cb_prob = cb_predict(feat, cb)

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric('Aggression %', f"{feat['aggression_ratio']*100:.0f}%")
            m2.metric('Intent-to-harm', f'{max_intent*100:.0f}%')
            m3.metric('Cyberbullying prob.', f'{cb_prob*100:.1f}%')
            if cb_prob >= 0.5:
                st.error('⚠️ CYBERBULLYING detected in this relationship')
            else:
                st.success('✅ No cyberbullying pattern detected')

# ── Tab 3: meme (image + text) ──────────────────────────────────────────────
with tab3:
    st.subheader('Analyse a meme (image + text)')
    im = load_image_model()
    if im is None:
        st.warning('Image model not found. Place `models/image_fusion_mbert.pth` to enable this tab.')
    else:
        itok, imodel, itf = im
        up = st.file_uploader('Meme image', type=['jpg', 'jpeg', 'png'])
        mtext = st.text_input('Meme text (OCR / caption)', '')
        if st.button('Analyse meme', type='primary') and up is not None:
            from PIL import Image
            img = Image.open(up).convert('RGB')
            st.image(img, width=280)
            enc = itok(PRE.preprocess(mtext, lang='en'), add_special_tokens=True,
                       max_length=MAX_LEN, padding='max_length', truncation=True,
                       return_token_type_ids=True, return_tensors='pt')
            it = itf(img).unsqueeze(0).to(DEVICE)
            ctx = torch.zeros(1, 2).to(DEVICE)
            with torch.no_grad():
                lo = imodel(enc['input_ids'].to(DEVICE), enc['attention_mask'].to(DEVICE),
                            enc['token_type_ids'].to(DEVICE), it, ctx).squeeze(-1)
                p = float(torch.sigmoid(lo).item())
            st.metric('Offensive probability', f'{p*100:.1f}%')
            if p >= 0.5:
                st.error('⚠️ Offensive meme')
            else:
                st.success('✅ Not offensive')
            st.caption('Note: the multimodal meme model is a weak baseline (thesis §4.x) — '
                       'shown to demonstrate the image+text pipeline, not for reliable use.')

st.divider()
st.caption('FYP — Context-Aware Multilingual Cyberbullying Detection · m-BERT / MuRIL + dense fusion')
