import streamlit as st
import base64
import json
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from openai import OpenAI

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CaptionAI — Marketing Caption Generator",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=Cairo:wght@400;600;700&display=swap');

:root {
    --ink:       #0D0D0D;
    --paper:     #F7F5F0;
    --accent:    #FF4D00;
    --accent2:   #1A1AFF;
    --muted:     #888;
    --border:    #E0DDD6;
    --card:      #FFFFFF;
    --success:   #00C06E;
    --warn:      #FFB800;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', 'Cairo', sans-serif;
    background: var(--paper);
    color: var(--ink);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* RTL / LTR */
.rtl { direction: rtl; text-align: right; }
.ltr { direction: ltr; text-align: left; }

/* ── HERO HEADER ── */
.hero {
    background: var(--ink);
    color: #fff;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "✦";
    position: absolute;
    right: 2rem; top: 1rem;
    font-size: 80px;
    opacity: 0.06;
    font-family: 'Syne', sans-serif;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 0 0 .3rem;
    letter-spacing: -1px;
}
.hero p { margin: 0; opacity: .65; font-size: 14px; }

/* ── STEP BADGE ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--ink);
    color: #fff;
    font-family: 'Syne', sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 4px 12px;
    border-radius: 100px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── CAPTION CARD ── */
.caption-card {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    min-height: 160px;
    line-height: 1.85;
    font-size: 15px;
    transition: box-shadow .2s;
}
.caption-card:hover { box-shadow: 0 6px 24px rgba(0,0,0,.08); }

.caption-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 10px;
}

.copy-btn-wrapper { margin-top: 12px; }

/* ── AI JUDGE CARD ── */
.judge-card {
    background: linear-gradient(135deg, #0D0D0D 0%, #1a1a2e 100%);
    color: #fff;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.judge-card h4 {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #FF4D00;
    margin: 0 0 1rem;
}
.judge-score-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.judge-criterion { font-size: 13px; opacity: .75; width: 130px; flex-shrink: 0; }
.judge-bar-bg {
    flex: 1;
    background: rgba(255,255,255,.1);
    border-radius: 100px;
    height: 8px;
    overflow: hidden;
}
.judge-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #FF4D00, #FFB800);
    transition: width .6s ease;
}
.judge-score-num { font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700; width: 30px; text-align: right; }
.judge-feedback {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,.1);
    font-size: 13px;
    opacity: .8;
    line-height: 1.7;
}

/* ── RATING AREA ── */
.rating-section {
    background: var(--card);
    border: 1.5px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.rating-title {
    font-family: 'Syne', sans-serif;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── METRIC PILL ── */
.metric-pill {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: .6rem 1rem;
    min-width: 90px;
}
.metric-pill-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1;
}
.metric-pill-label { font-size: 10px; color: var(--muted); letter-spacing: 1px; text-transform: uppercase; margin-top: 4px; }

/* ── WINNER BANNER ── */
.winner-banner {
    background: var(--success);
    color: #fff;
    border-radius: 12px;
    padding: .9rem 1.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}
.tie-banner {
    background: var(--warn);
    color: #fff;
    border-radius: 12px;
    padding: .9rem 1.4rem;
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
}

/* ── PROMPT PREVIEW ── */
.prompt-box {
    background: #F0EDE6;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 13px;
    font-family: 'DM Mono', monospace;
    white-space: pre-wrap;
    border-left: 3px solid var(--accent);
    line-height: 1.7;
    color: #333;
}

/* ── FEEDBACK HISTORY ── */
.feedback-item {
    border-left: 3px solid var(--accent2);
    padding: .6rem 1rem;
    margin-bottom: .6rem;
    background: #f8f8ff;
    border-radius: 0 10px 10px 0;
    font-size: 13px;
    line-height: 1.6;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--ink) !important;
}
section[data-testid="stSidebar"] * {
    color: #fff !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb] {
    background: rgba(255,255,255,.08) !important;
    border-color: rgba(255,255,255,.15) !important;
    color: #fff !important;
    border-radius: 10px !important;
}
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.1) !important; }
section[data-testid="stSidebar"] .stRadio label { color: #ccc !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #fff;
    margin-bottom: .2rem;
}
.sidebar-sub { font-size: 11px; color: rgba(255,255,255,.4); letter-spacing: 1px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ─── i18n ────────────────────────────────────────────────────────────────────
T = {
    "en": {
        "hero_title": "✦ CaptionAI",
        "hero_sub": "Generate & compare professional marketing captions powered by GPT-4o Vision",
        "lang_label": "Caption Language",
        "apikey_label": "OpenAI API Key",
        "apikey_help": "Key is used in-session only — never stored.",
        "tab_gen": "✦ Generate",
        "tab_analysis": "◎ Analysis",
        "tab_batch": "⊞ Batch",
        "tab_export": "↓ Export",
        "step1": "Step 1 — Upload Image",
        "step2": "Step 2 — Product Details",
        "step3": "Step 3 — Generate",
        "upload_prompt": "Drop your product image here (PNG / JPG / WEBP)",
        "product_label": "Product Name",
        "audience_label": "Target Audience",
        "tone_label": "Tone",
        "platform_label": "Platform",
        "usp_label": "Unique Selling Point",
        "usp_help": "What makes this product special?",
        "prompt_preview": "Prompt Preview",
        "generate_btn": "▶  Generate Caption",
        "caption_label": "Generated Caption",
        "copy_btn": "Copy Caption",
        "copied": "✓ Copied to clipboard!",
        "ai_judge_title": "AI Judge — Auto Evaluation",
        "ai_judge_btn": "⚡ Ask AI to Evaluate",
        "rate_title": "Your Rating",
        "criteria": ["Persuasiveness", "Professionalism", "Audience Fit", "Creativity"],
        "save_rating": "Save Rating",
        "rating_saved": "Rating saved!",
        "analysis_header": "Results & Analysis",
        "no_ratings": "Generate a caption and save a rating first.",
        "avg_score": "Avg Score",
        "feedback_history": "AI Feedback History",
        "export_header": "Export Results",
        "export_json": "Download JSON",
        "export_csv": "Download CSV",
        "tones": ["Persuasive", "Professional", "Casual", "Luxury", "Energetic", "Witty"],
        "platforms": ["Instagram", "Facebook", "LinkedIn", "TikTok", "Twitter/X", "General"],
        "err_nokey": "⚠ Enter your OpenAI API key in the sidebar.",
        "err_noimg": "⚠ Upload a product image first.",
        "batch_header": "Batch Generation",
        "batch_empty": "Add at least one product to run.",
        "batch_add": "Add Product",
        "batch_run": "▶  Run Batch",
        "winner": "Better Prompt",
        "feedback_every": "📌 AI Feedback milestone reached!",
    },
    "ar": {
        "hero_title": "✦ CaptionAI",
        "hero_sub": "أنشئ وقارن تعليقات تسويقية احترافية بقوة GPT-4o Vision",
        "lang_label": "لغة التعليق",
        "apikey_label": "مفتاح OpenAI API",
        "apikey_help": "المفتاح يُستخدم في الجلسة فقط ولا يُخزَّن.",
        "tab_gen": "✦ توليد",
        "tab_analysis": "◎ تحليل",
        "tab_batch": "⊞ دفعة",
        "tab_export": "↓ تصدير",
        "step1": "الخطوة ١ — رفع الصورة",
        "step2": "الخطوة ٢ — تفاصيل المنتج",
        "step3": "الخطوة ٣ — التوليد",
        "upload_prompt": "ارفع صورة المنتج هنا (PNG / JPG / WEBP)",
        "product_label": "اسم المنتج",
        "audience_label": "الجمهور المستهدف",
        "tone_label": "الأسلوب",
        "platform_label": "المنصة",
        "usp_label": "نقطة البيع الفريدة",
        "usp_help": "ما الذي يميّز هذا المنتج؟",
        "prompt_preview": "معاينة الموجّه",
        "generate_btn": "▶  توليد التعليق",
        "caption_label": "التعليق المُولَّد",
        "copy_btn": "نسخ التعليق",
        "copied": "✓ تم النسخ!",
        "ai_judge_title": "المحكّم الذكي — تقييم تلقائي",
        "ai_judge_btn": "⚡ اطلب تقييم الذكاء الاصطناعي",
        "rate_title": "تقييمك",
        "criteria": ["الإقناع", "الاحترافية", "توافق الجمهور", "الإبداع"],
        "save_rating": "حفظ التقييم",
        "rating_saved": "تم حفظ التقييم!",
        "analysis_header": "النتائج والتحليل",
        "no_ratings": "قم بتوليد تعليق وحفظ تقييم أولاً.",
        "avg_score": "متوسط الدرجات",
        "feedback_history": "سجل ملاحظات الذكاء الاصطناعي",
        "export_header": "تصدير النتائج",
        "export_json": "تنزيل JSON",
        "export_csv": "تنزيل CSV",
        "tones": ["مقنع", "احترافي", "غير رسمي", "فاخر", "نشيط", "طريف"],
        "platforms": ["Instagram", "Facebook", "LinkedIn", "TikTok", "Twitter/X", "عام"],
        "err_nokey": "⚠ أدخل مفتاح OpenAI API في الشريط الجانبي.",
        "err_noimg": "⚠ ارفع صورة المنتج أولاً.",
        "batch_header": "التوليد الدفعي",
        "batch_empty": "أضف منتجاً واحداً على الأقل.",
        "batch_add": "إضافة منتج",
        "batch_run": "▶  تشغيل الدفعة",
        "winner": "الموجّه الأفضل",
        "feedback_every": "📌 تم الوصول إلى نقطة تغذية راجعة!",
    },
}

def t(key):
    return T[st.session_state.get("lang", "en")][key]

# ─── Session state ────────────────────────────────────────────────────────────
defaults = {
    "lang": "en",
    "caption": "",
    "image_b64": None,
    "image_mime": "image/jpeg",
    "all_ratings": [],       # list of dicts with user scores + ai scores
    "ai_feedbacks": [],      # list of AI feedback strings
    "current_ai_eval": None, # dict: {scores: [...], feedback: str}
    "current_user_scores": [],
    "batch_products": [],
    "batch_results": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers ─────────────────────────────────────────────────────────────────
def image_to_b64(img_bytes):
    return base64.b64encode(img_bytes).decode("utf-8")


def build_prompt(params: dict, lang: str) -> str:
    """Single, powerful structured prompt."""
    usp_line = f"\n- Unique Selling Point: {params['usp']}" if params.get("usp") else ""
    if lang == "ar":
        usp_line = f"\n- نقطة البيع الفريدة: {params['usp']}" if params.get("usp") else ""
        return (
            "أنت خبير تسويق رقمي محترف ومبدع. مهمتك كتابة تعليق تسويقي استثنائي باللغة العربية.\n\n"
            "📌 تعليمات:\n"
            "- ابدأ بجملة افتتاحية قوية تشد الانتباه فوراً\n"
            "- ركّز على الفائدة العاطفية وليس المواصفات فقط\n"
            "- استخدم صيغة الخطاب المباشر للجمهور المحدد\n"
            "- أنهِ بدعوة واضحة وجذابة للتصرف (CTA) تناسب المنصة\n"
            "- لا تتجاوز ٥ أسطر\n\n"
            f"🎯 بيانات الحملة:\n"
            f"- المنتج: {params['product_name']}\n"
            f"- الجمهور: {params['target_audience']}\n"
            f"- الأسلوب: {params['tone']}\n"
            f"- المنصة: {params['platform']}"
            f"{usp_line}\n\n"
            "اكتب التعليق الآن مباشرةً دون مقدمات:"
        )
    return (
        "You are a world-class digital marketing copywriter. Your task is to write an exceptional marketing caption.\n\n"
        "📌 Instructions:\n"
        "- Open with a powerful hook that grabs attention instantly\n"
        "- Focus on emotional benefit, not just specs\n"
        "- Address the target audience directly\n"
        "- Close with a compelling, platform-appropriate CTA\n"
        "- Maximum 5 lines\n\n"
        f"🎯 Campaign Details:\n"
        f"- Product: {params['product_name']}\n"
        f"- Target Audience: {params['target_audience']}\n"
        f"- Tone: {params['tone']}\n"
        f"- Platform: {params['platform']}"
        f"{usp_line}\n\n"
        "Write the caption now, directly, without preamble:"
    )


def generate_caption(client, prompt, b64, mime) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return resp.choices[0].message.content.strip()


def ai_evaluate_caption(client, caption: str, params: dict, lang: str) -> dict:
    """AI judge evaluates the caption and returns scores + textual feedback."""
    criteria_en = ["Persuasiveness", "Professionalism", "Audience Fit", "Creativity"]
    criteria_ar = ["الإقناع", "الاحترافية", "توافق الجمهور", "الإبداع"]
    criteria = criteria_ar if lang == "ar" else criteria_en

    system = (
        "You are an expert marketing evaluator. "
        "Evaluate the given marketing caption strictly and objectively. "
        "Respond ONLY with valid JSON, no markdown, no extra text."
    )
    user_msg = (
        f"Evaluate this marketing caption for a {params.get('product_name','product')} "
        f"targeting {params.get('target_audience','general audience')} "
        f"on {params.get('platform','general')} platform.\n\n"
        f"Caption:\n{caption}\n\n"
        f"Score each criterion from 1 to 5 (integers only). "
        f"Also provide a short feedback string (2-3 sentences, constructive).\n"
        f"Respond with this exact JSON structure:\n"
        '{"scores": {"Persuasiveness": X, "Professionalism": X, "Audience Fit": X, "Creativity": X}, '
        '"feedback": "..."}'
    )
    resp = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=300,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    # Strip markdown fences if any
    raw = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(raw)
    # Remap keys to current language if Arabic
    if lang == "ar":
        mapped = {}
        key_map = dict(zip(criteria_en, criteria_ar))
        for k, v in data["scores"].items():
            mapped[key_map.get(k, k)] = v
        data["scores"] = mapped
    return data


def bar_chart(user_scores, ai_scores, criteria):
    x = np.arange(len(criteria))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - w/2, user_scores, w, label="You", color="#0D0D0D", alpha=0.9)
    ax.bar(x + w/2, ai_scores, w, label="AI Judge", color="#FF4D00", alpha=0.9)
    for xi, v in zip(x - w/2, user_scores):
        ax.text(xi, v + 0.08, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold")
    for xi, v in zip(x + w/2, ai_scores):
        ax.text(xi, v + 0.08, str(v), ha="center", va="bottom", fontsize=9, fontweight="bold", color="#FF4D00")
    ax.set_xticks(x)
    ax.set_xticklabels(criteria, fontsize=9)
    ax.set_ylim(0, 6)
    ax.set_ylabel("Score (1–5)", fontsize=9)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("#F7F5F0")
    fig.patch.set_facecolor("#F7F5F0")
    fig.tight_layout()
    return fig


def radar_chart(user_scores, ai_scores, criteria):
    N = len(criteria)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    u = user_scores + user_scores[:1]
    a = ai_scores + ai_scores[:1]
    ang = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(ang, u, "o-", lw=2, color="#0D0D0D", label="You")
    ax.fill(ang, u, alpha=0.12, color="#0D0D0D")
    ax.plot(ang, a, "o-", lw=2, color="#FF4D00", label="AI Judge")
    ax.fill(ang, a, alpha=0.12, color="#FF4D00")
    ax.set_xticks(angles)
    ax.set_xticklabels(criteria, size=8)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1","2","3","4","5"], size=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    ax.set_facecolor("#F7F5F0")
    fig.patch.set_facecolor("#F7F5F0")
    fig.tight_layout()
    return fig


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">✦ CaptionAI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Marketing Caption Generator</div>', unsafe_allow_html=True)
    st.divider()

    lang_choice = st.radio(t("lang_label"), ["English", "العربية"],
                           index=0 if st.session_state.lang == "en" else 1,
                           horizontal=True)
    st.session_state.lang = "en" if lang_choice == "English" else "ar"

    api_key = st.text_input(t("apikey_label"), type="password", help=t("apikey_help"))

    st.divider()
    st.caption("Model: GPT-4o Vision")
    st.caption("Evaluator: GPT-4o (AI Judge)")
    st.caption("Scale: 1–5 Likert")

    # Rating counter
    n = len(st.session_state.all_ratings)
    st.divider()
    st.markdown(f"**{n}** ratings saved")
    if n > 0:
        all_user = [r["user_avg"] for r in st.session_state.all_ratings]
        all_ai   = [r["ai_avg"]   for r in st.session_state.all_ratings if r.get("ai_avg")]
        st.caption(f"Your avg: **{sum(all_user)/len(all_user):.1f}**")
        if all_ai:
            st.caption(f"AI avg: **{sum(all_ai)/len(all_ai):.1f}**")

# ─── Hero ─────────────────────────────────────────────────────────────────────
rtl = "rtl" if st.session_state.lang == "ar" else "ltr"
st.markdown(
    f'<div class="hero {rtl}"><h1>{t("hero_title")}</h1><p>{t("hero_sub")}</p></div>',
    unsafe_allow_html=True
)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([t("tab_gen"), t("tab_analysis"), t("tab_batch"), t("tab_export")])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([1, 1.4], gap="large")

    # ── LEFT: image + params ──────────────────────────────────────────────────
    with col_l:
        st.markdown(f'<div class="step-badge">{"Step 1" if st.session_state.lang=="en" else "الخطوة ١"} — {"Upload Image" if st.session_state.lang=="en" else "رفع الصورة"}</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            t("upload_prompt"),
            type=["png","jpg","jpeg","webp"],
            label_visibility="visible",
        )
        if uploaded:
            raw = uploaded.read()
            ext = uploaded.name.rsplit(".",1)[-1].lower()
            mime_map = {"png":"image/png","jpg":"image/jpeg","jpeg":"image/jpeg","webp":"image/webp"}
            st.session_state.image_mime = mime_map.get(ext, "image/jpeg")
            st.session_state.image_b64 = image_to_b64(raw)
            st.image(Image.open(io.BytesIO(raw)), use_container_width=True)

        st.divider()
        st.markdown(f'<div class="step-badge">{"Step 2" if st.session_state.lang=="en" else "الخطوة ٢"} — {"Product Details" if st.session_state.lang=="en" else "تفاصيل المنتج"}</div>', unsafe_allow_html=True)

        product_name    = st.text_input(t("product_label"), placeholder="e.g. AirPods Pro" if st.session_state.lang=="en" else "مثال: سماعات AirPods")
        target_audience = st.text_input(t("audience_label"), placeholder="e.g. Young professionals" if st.session_state.lang=="en" else "مثال: المهنيون الشباب")
        c1, c2 = st.columns(2)
        with c1: tone     = st.selectbox(t("tone_label"), t("tones"))
        with c2: platform = st.selectbox(t("platform_label"), t("platforms"))
        usp = st.text_input(t("usp_label"), help=t("usp_help"))

    # ── RIGHT: prompt + generate + result ────────────────────────────────────
    with col_r:
        params = {
            "product_name":    product_name    or ("the product"          if st.session_state.lang=="en" else "المنتج"),
            "target_audience": target_audience or ("potential customers"   if st.session_state.lang=="en" else "العملاء المحتملين"),
            "tone":     tone,
            "platform": platform,
            "usp":      usp,
        }
        prompt = build_prompt(params, st.session_state.lang)

        with st.expander(f"🔍 {t('prompt_preview')}"):
            st.markdown(f'<div class="prompt-box {rtl}">{prompt}</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="step-badge">{"Step 3" if st.session_state.lang=="en" else "الخطوة ٣"} — {"Generate" if st.session_state.lang=="en" else "التوليد"}</div>', unsafe_allow_html=True)

        if st.button(t("generate_btn"), use_container_width=True, type="primary"):
            if not api_key:
                st.error(t("err_nokey"))
            elif not st.session_state.image_b64:
                st.error(t("err_noimg"))
            else:
                with st.spinner("Generating..."):
                    try:
                        client = OpenAI(api_key=api_key)
                        st.session_state.caption = generate_caption(
                            client, prompt,
                            st.session_state.image_b64,
                            st.session_state.image_mime
                        )
                        st.session_state.current_ai_eval = None  # reset judge
                    except Exception as e:
                        st.error(f"Error: {e}")

        # ── Caption Display ──
        if st.session_state.caption:
            caption_text = st.session_state.caption
            import html as html_lib
            import streamlit.components.v1 as components
            caption_escaped = html_lib.escape(caption_text)
            generated_label = "GENERATED CAPTION" if st.session_state.lang == "en" else "التعليق المُولَّد"
            copy_label   = "📋 Copy Caption" if st.session_state.lang == "en" else "📋 نسخ التعليق"
            copied_label = "✓ Copied!"       if st.session_state.lang == "en" else "✓ تم النسخ!"

            # Caption card (plain HTML, no JS needed here)
            st.markdown(
                f'<div class="caption-label {rtl}">{generated_label}</div>'
                f'<div class="caption-card {rtl}">{caption_escaped}</div>',
                unsafe_allow_html=True,
            )

            # Copy button inside st.components.v1.html — has its own iframe
            # context so clipboard API + execCommand fallback both work
            components.html(f"""
            <button id="cpyBtn" onclick="copyIt()" style="
                background:#0D0D0D;color:#fff;border:none;border-radius:8px;
                padding:9px 22px;font-size:13px;font-weight:600;cursor:pointer;
                font-family:sans-serif;letter-spacing:.5px;margin-top:6px;">
                {copy_label}
            </button>
            <script>
            const TEXT = {json.dumps(caption_text)};
            function copyIt() {{
                var btn = document.getElementById('cpyBtn');
                function ok() {{
                    btn.innerText = '{copied_label}';
                    btn.style.background = '#00C06E';
                    setTimeout(function(){{
                        btn.innerText = '{copy_label}';
                        btn.style.background = '#0D0D0D';
                    }}, 2000);
                }}
                function fallback() {{
                    var ta = document.createElement('textarea');
                    ta.value = TEXT;
                    ta.style.position = 'fixed';
                    ta.style.opacity = '0';
                    document.body.appendChild(ta);
                    ta.focus(); ta.select();
                    try {{ document.execCommand('copy'); ok(); }}
                    catch(e) {{ btn.innerText = '⚠ Select text manually'; }}
                    document.body.removeChild(ta);
                }}
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(TEXT).then(ok, fallback);
                }} else {{
                    fallback();
                }}
            }}
            </script>
            """, height=55)

            st.divider()

            # ── AI JUDGE ──────────────────────────────────────────────────
            st.markdown(f'<div class="step-badge" style="background:#FF4D00;">⚡ {"AI Judge" if st.session_state.lang=="en" else "المحكّم الذكي"}</div>', unsafe_allow_html=True)

            if st.button(t("ai_judge_btn"), use_container_width=True):
                if not api_key:
                    st.error(t("err_nokey"))
                else:
                    with st.spinner("Evaluating..." if st.session_state.lang=="en" else "جارٍ التقييم..."):
                        try:
                            client = OpenAI(api_key=api_key)
                            result = ai_evaluate_caption(client, caption_text, params, st.session_state.lang)
                            st.session_state.current_ai_eval = result
                            # Save to feedback history
                            st.session_state.ai_feedbacks.append(result["feedback"])
                        except Exception as e:
                            st.error(f"Evaluation error: {e}")

            if st.session_state.current_ai_eval:
                ev = st.session_state.current_ai_eval
                scores_dict = ev["scores"]
                criteria = t("criteria")
                scores_list = [scores_dict.get(c, 3) for c in criteria]
                avg = round(sum(scores_list)/len(scores_list), 1)

                # Render judge card using native Streamlit (avoids HTML escape issues)
                with st.container():
                    st.markdown(
                        f'<div style="background:linear-gradient(135deg,#0D0D0D,#1a1a2e);'
                        f'border-radius:16px;padding:1.5rem;margin-top:.5rem;">'
                        f'<span style="font-family:Syne,sans-serif;font-size:11px;letter-spacing:2px;'
                        f'text-transform:uppercase;color:#FF4D00;font-weight:700;">'
                        f'{t("ai_judge_title")}</span>'
                        f'&nbsp;&nbsp;<span style="font-family:Syne,sans-serif;font-size:1.6rem;'
                        f'font-weight:800;color:#FFB800;">{avg}</span>'
                        f'<span style="color:#888;font-size:13px;">/5</span></div>',
                        unsafe_allow_html=True,
                    )
                    # Score bars — one row per criterion using native progress + columns
                    for crit, sc in zip(criteria, scores_list):
                        c_left, c_bar, c_num = st.columns([1.5, 4, 0.6])
                        with c_left:
                            st.markdown(f'<span style="font-size:13px;color:#555;">{crit}</span>', unsafe_allow_html=True)
                        with c_bar:
                            st.progress(sc / 5)
                        with c_num:
                            st.markdown(f'<span style="font-family:Syne,sans-serif;font-weight:700;font-size:14px;">{sc}/5</span>', unsafe_allow_html=True)

                    # Feedback text
                    st.markdown(
                        f'<div style="margin-top:.8rem;padding:.8rem 1rem;'
                        f'background:rgba(255,255,255,.06);border-radius:10px;'
                        f'font-size:13px;color:#ccc;line-height:1.7;direction:{"rtl" if st.session_state.lang=="ar" else "ltr"};">'
                        f'{html_lib.escape(ev["feedback"])}</div>',
                        unsafe_allow_html=True,
                    )

            st.divider()

            # ── USER RATING ───────────────────────────────────────────────
            st.markdown(f'<div class="step-badge" style="background:#1A1AFF;">★ {"Your Rating" if st.session_state.lang=="en" else "تقييمك"}</div>', unsafe_allow_html=True)

            criteria = t("criteria")
            user_scores = []
            rc = st.columns(len(criteria))
            for i, (crit, col) in enumerate(zip(criteria, rc)):
                with col:
                    st.caption(crit)
                    v = st.slider(crit, 1, 5, 3, key=f"user_{i}", label_visibility="collapsed")
                    user_scores.append(v)

            if st.button(t("save_rating"), use_container_width=True):
                ai_eval = st.session_state.current_ai_eval
                ai_scores_list = []
                if ai_eval:
                    ai_scores_list = [ai_eval["scores"].get(c, 0) for c in criteria]

                entry = {
                    "product":     product_name,
                    "platform":    platform,
                    "caption":     caption_text,
                    "user_scores": user_scores,
                    "user_avg":    round(sum(user_scores)/len(user_scores), 2),
                    "ai_scores":   ai_scores_list,
                    "ai_avg":      round(sum(ai_scores_list)/len(ai_scores_list), 2) if ai_scores_list else None,
                    "ai_feedback": ai_eval["feedback"] if ai_eval else "",
                }
                st.session_state.all_ratings.append(entry)
                st.session_state.current_user_scores = user_scores
                st.success(t("rating_saved"))

                # Milestone feedback every 3 ratings
                n = len(st.session_state.all_ratings)
                if n % 3 == 0 and n > 0:
                    st.info(t("feedback_every"))
                    if api_key:
                        past = st.session_state.all_ratings[-3:]
                        summary = "\n".join([
                            f"Caption: {r['caption'][:80]}... | User: {r['user_avg']} | AI: {r.get('ai_avg','N/A')}"
                            for r in past
                        ])
                        with st.spinner("Generating milestone feedback..."):
                            try:
                                client = OpenAI(api_key=api_key)
                                mile_resp = client.chat.completions.create(
                                    model="gpt-4o", max_tokens=200,
                                    messages=[{"role":"user","content":
                                        f"Based on these 3 caption evaluations, give a brief strategic insight about patterns in the ratings and how to improve future captions:\n\n{summary}"}]
                                )
                                mile_text = mile_resp.choices[0].message.content.strip()
                                st.session_state.ai_feedbacks.append(f"[Milestone #{n//3}] {mile_text}")
                                st.success(f"💡 {mile_text}")
                            except:
                                pass


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(t("analysis_header"))

    if not st.session_state.all_ratings:
        st.info(t("no_ratings"))
    else:
        latest = st.session_state.all_ratings[-1]
        criteria = t("criteria")
        user_scores = latest["user_scores"]
        ai_scores   = latest.get("ai_scores", [3]*4)
        if not ai_scores:
            ai_scores = [3]*4

        # ── Metric pills ──
        cols = st.columns(len(criteria) + 2)
        for i, (crit, us, ai) in enumerate(zip(criteria, user_scores, ai_scores)):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-pill">
                    <div class="metric-pill-val">{us}<span style="font-size:12px;color:#999">/5</span></div>
                    <div class="metric-pill-label">{crit[:10]}</div>
                </div>""", unsafe_allow_html=True)
        with cols[-2]:
            st.markdown(f"""
            <div class="metric-pill" style="border-color:#0D0D0D">
                <div class="metric-pill-val" style="color:#0D0D0D">{latest['user_avg']}</div>
                <div class="metric-pill-label">You Avg</div>
            </div>""", unsafe_allow_html=True)
        with cols[-1]:
            ai_avg_val = latest.get("ai_avg","—")
            st.markdown(f"""
            <div class="metric-pill" style="border-color:#FF4D00">
                <div class="metric-pill-val" style="color:#FF4D00">{ai_avg_val}</div>
                <div class="metric-pill-label">AI Avg</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── Charts ──
        ch1, ch2 = st.columns([1.5, 1])
        with ch1:
            st.pyplot(bar_chart(user_scores, ai_scores, criteria), use_container_width=True)
        with ch2:
            st.pyplot(radar_chart(user_scores, ai_scores, criteria), use_container_width=True)

        st.divider()

        # ── All sessions table ──
        if len(st.session_state.all_ratings) > 0:
            rows = []
            for r in st.session_state.all_ratings:
                rows.append({
                    "Product": r.get("product",""),
                    "Platform": r.get("platform",""),
                    "Caption (preview)": r.get("caption","")[:60]+"...",
                    "Your Avg": r["user_avg"],
                    "AI Avg": r.get("ai_avg","—"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── AI Feedback History ──
        if st.session_state.ai_feedbacks:
            st.divider()
            st.subheader(t("feedback_history"))
            for fb in reversed(st.session_state.ai_feedbacks[-5:]):
                st.markdown(f'<div class="feedback-item {rtl}">{fb}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader(t("batch_header"))

    with st.expander(f"➕ {t('batch_add')}", expanded=not bool(st.session_state.batch_products)):
        bc1, bc2 = st.columns(2)
        with bc1:
            b_name     = st.text_input("Product Name", key="b_name")
            b_audience = st.text_input("Target Audience", key="b_audience")
            b_usp      = st.text_input("USP", key="b_usp")
        with bc2:
            b_tone     = st.selectbox("Tone", t("tones"), key="b_tone")
            b_platform = st.selectbox("Platform", t("platforms"), key="b_platform")
            b_url      = st.text_input("Image URL", key="b_url", placeholder="https://images.unsplash.com/...")

        if st.button("Add to Batch"):
            if b_name and b_url:
                st.session_state.batch_products.append({
                    "product_name": b_name, "target_audience": b_audience,
                    "tone": b_tone, "platform": b_platform, "usp": b_usp, "image_url": b_url,
                })
                st.success(f"Added: {b_name}")
            else:
                st.warning("Product name and image URL required.")

    if st.session_state.batch_products:
        st.dataframe(
            pd.DataFrame(st.session_state.batch_products)[["product_name","platform","tone","usp"]],
            use_container_width=True, hide_index=True,
        )
        if st.button(t("batch_run"), type="primary", use_container_width=True):
            if not api_key:
                st.error(t("err_nokey"))
            else:
                client = OpenAI(api_key=api_key)
                results = []
                prog = st.progress(0)
                status = st.empty()
                for i, prod in enumerate(st.session_state.batch_products):
                    status.text(f"Processing {i+1}/{len(st.session_state.batch_products)}: {prod['product_name']}...")
                    try:
                        resp = requests.get(prod["image_url"], timeout=10)
                        mime = resp.headers.get("Content-Type","image/jpeg").split(";")[0]
                        b64  = base64.b64encode(resp.content).decode()
                        cap  = generate_caption(client, build_prompt(prod, st.session_state.lang), b64, mime)
                        results.append({"Product": prod["product_name"], "Platform": prod["platform"], "Caption": cap})
                    except Exception as e:
                        results.append({"Product": prod["product_name"], "Platform": prod["platform"], "Caption": f"ERROR: {e}"})
                    prog.progress((i+1)/len(st.session_state.batch_products))
                status.text("✅ Done!")
                st.session_state.batch_results = results

        if st.session_state.batch_results:
            df_b = pd.DataFrame(st.session_state.batch_results)
            st.dataframe(df_b, use_container_width=True, hide_index=True)
            st.download_button("⬇ Download CSV", df_b.to_csv(index=False,encoding="utf-8-sig"), "batch.csv","text/csv")
    else:
        st.info(t("batch_empty"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader(t("export_header"))

    if not st.session_state.all_ratings:
        st.info(t("no_ratings"))
    else:
        export = {
            "ratings": st.session_state.all_ratings,
            "ai_feedbacks": st.session_state.ai_feedbacks,
        }
        json_str = json.dumps(export, ensure_ascii=False, indent=2)
        st.code(json_str, language="json")
        st.download_button(t("export_json"), json_str, "results.json", "application/json", use_container_width=True)

        rows = []
        criteria = t("criteria")
        for r in st.session_state.all_ratings:
            row = {"Product": r.get("product",""), "Platform": r.get("platform",""),
                   "Caption": r.get("caption",""), "User_Avg": r["user_avg"], "AI_Avg": r.get("ai_avg",""),
                   "AI_Feedback": r.get("ai_feedback","")}
            for c, s in zip(criteria, r.get("user_scores",[])):
                row[f"User_{c}"] = s
            for c, s in zip(criteria, r.get("ai_scores",[])):
                row[f"AI_{c}"] = s
            rows.append(row)
        df_all = pd.DataFrame(rows)
        st.download_button(t("export_csv"), df_all.to_csv(index=False,encoding="utf-8-sig"), "ratings.csv","text/csv", use_container_width=True)
