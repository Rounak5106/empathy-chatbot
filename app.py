import streamlit as st
from transformers import pipeline
import random
import re

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="EmpathAI – Mental Health Support",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .emotion-pill {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 500;
        margin-top: 6px;
    }
    .crisis-box {
        background: #2d1515;
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 14px 18px;
        margin-top: 8px;
        font-size: 14px;
        line-height: 1.8;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  CONSTANTS
# ──────────────────────────────────────────────

EMOTION_COLORS = {
    "sadness":  "#4a90d9",
    "fear":     "#9b59b6",
    "anger":    "#e74c3c",
    "joy":      "#2ecc71",
    "disgust":  "#e67e22",
    "surprise": "#f39c12",
    "neutral":  "#7f8c8d",
}

CONFIDENCE_THRESHOLD = 0.50   # raised slightly — more conservative

# ── Negation patterns ──────────────────────────
# If any of these match, flip joy→sadness, flip high-confidence positive→neutral
# Format: (regex_pattern, forced_emotion)
NEGATION_RULES = [
    # "not feeling well / good / great / fine / okay"
    (r"\bnot feeling (well|good|great|fine|okay|alright|better)\b", "sadness"),
    # "don't feel well / good"
    (r"\bdon'?t feel (well|good|great|fine|okay|alright)\b", "sadness"),
    # "not feeling like myself / anything"
    (r"\bnot feeling (like myself|anything|much)\b", "sadness"),
    # "i'm not okay / not fine / not great / not alright"
    (r"\bi'?m not (okay|fine|great|alright|good|well)\b", "sadness"),
    # "not doing well / good"
    (r"\bnot doing (well|good|great|okay|fine)\b", "sadness"),
    # "can't feel anything / feel nothing"
    (r"\b(can'?t feel|feel nothing|feeling nothing)\b", "sadness"),
    # "don't feel happy / joyful"
    (r"\bdon'?t feel (happy|joyful|joy|excited|glad)\b", "sadness"),
    # "not happy / not joyful"
    (r"\bnot (happy|joyful|glad|excited|cheerful)\b", "sadness"),
    # "nothing feels good / right / meaningful"
    (r"\bnothing feels (good|right|meaningful|okay|fine|real)\b", "sadness"),
    # "no longer happy / excited"
    (r"\bno longer (happy|excited|glad|joyful|fine|okay)\b", "sadness"),
]

# ── Crisis keywords ────────────────────────────
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "don't want to be here", "no reason to live", "self harm",
    "hurt myself", "cutting myself", "overdose", "jump off",
    "hang myself", "i want to disappear forever", "not worth living",
    "better off dead", "can't go on",
]

CRISIS_RESPONSE = """I'm really glad you reached out. What you're feeling matters, and you don't have to face this alone.

🆘 **If you are in immediate danger, call emergency services: 112**

**Free helplines (India — available 24/7):**
- **iCall:** 9152987821
- **Vandrevala Foundation:** 1860-2662-345
- **AASRA:** 9820466627

Please reach out to one of these — a real person is ready to listen right now."""

# ── Keyword emotion map (fallback) ────────────
KEYWORD_EMOTION_MAP = {
    "sadness": [
        "sad", "depressed", "unhappy", "miserable", "hopeless", "lonely",
        "grief", "crying", "heartbroken", "empty", "worthless", "numb",
        "meaningless", "nothing feels", "can't feel", "don't feel",
        "lost interest", "no motivation", "tired of", "exhausted",
        "not well", "unwell", "not okay", "not fine", "not good",
        "not feeling well", "don't feel well", "feeling low", "feeling down",
        "feel terrible", "feel awful", "feel horrible", "not myself",
    ],
    "fear": [
        "scared", "afraid", "anxious", "anxiety", "panic", "worried",
        "nervous", "terrified", "worst case", "imagining", "what if",
        "overthinking", "can't stop thinking", "racing thoughts",
        "won't stop", "keeps happening", "catastrophe", "disaster",
    ],
    "anger": [
        "angry", "furious", "irritated", "annoyed", "frustrated",
        "rage", "hate", "fed up", "sick of", "can't stand",
        "keeps irritating", "everyone irritates", "so unfair",
    ],
    "joy": [
        "happy", "excited", "great", "amazing", "wonderful",
        "fantastic", "joyful", "grateful", "blessed", "glad",
        "positive", "good day", "feeling good", "feeling well",
    ],
    "disgust": [
        "disgusted", "gross", "horrible", "repulsed", "revolting",
    ],
}

# ── CBT response templates ─────────────────────
# {snippet} is replaced with phrase from user's actual message
RESPONSES = {
    "sadness": [
        "It sounds like you're not feeling your best right now — '{snippet}' is a hard thing to sit with. That kind of feeling is valid, and it doesn't mean things will stay this way. Can you tell me a bit more about what's been going on?",
        "When you say '{snippet}', I hear that things feel heavy right now. In CBT, we often check whether our current state is being influenced by our thoughts — do you notice any particular thoughts that come up when you feel this way?",
        "'{snippet}' — that's a real and difficult experience. Sometimes when we feel this way, it helps to ask: is there one small thing, even tiny, that could make the next hour slightly more bearable? What comes to mind?",
    ],
    "fear": [
        "Imagining '{snippet}' puts your mind in a state of high alert — that's your brain trying to protect you, even if it's causing more harm than good. A CBT technique that helps: ask yourself what's the realistic probability of the worst case actually happening, and what evidence supports it?",
        "When our thoughts fixate on '{snippet}', it's often a sign of catastrophizing — where the mind fast-forwards to the worst outcome. Try grounding yourself: name 5 things you can see right now. Then let's talk about what's actually in your control here.",
        "'{snippet}' — that pattern of worry is exhausting. One CBT tool is scheduling a 'worry window' of 10 minutes per day where you allow yourself to think about fears, then consciously set them aside. Have you tried anything like that before?",
    ],
    "anger": [
        "Feeling frustrated about '{snippet}' makes complete sense — when we feel unheard or disrespected repeatedly, anger is a natural response. Before acting on it, can you identify the core need underneath? Often it's a need for respect, fairness, or control.",
        "'{snippet}' — that frustration is real. Anger often sits on top of a deeper emotion like hurt or helplessness. What do you think is underneath it in this case?",
        "When '{snippet}' keeps happening, it can feel like you have no control. A CBT approach is to separate what you can control from what you can't — and focus only on the former. What's one part of this situation you actually have influence over?",
    ],
    "joy": [
        "It's great to hear '{snippet}'! Positive moments are worth pausing on — our brains have a negativity bias, so consciously noticing good feelings actually helps reinforce them. What made this moment stand out for you?",
        "'{snippet}' — that sounds genuinely positive. How are you feeling about things overall right now?",
    ],
    "disgust": [
        "It sounds like '{snippet}' has left you feeling really uncomfortable. That reaction often signals that something violated a value you hold. What specifically felt wrong about it?",
    ],
    "surprise": [
        "'{snippet}' — sounds like something unexpected happened. Are you feeling more positive or unsettled about it?",
    ],
    "neutral": [
        "You mentioned '{snippet}'. I want to make sure I understand — can you tell me a bit more about how that's been making you feel?",
        "Thank you for sharing that. When you say '{snippet}', what emotion comes up most strongly for you right now?",
        "I hear you. '{snippet}' — sometimes it's hard to name exactly what we're feeling. Would you describe it more as heaviness, tension, numbness, or something else?",
    ],
}

# ──────────────────────────────────────────────
#  LOAD MODEL
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading emotion model… (first run takes ~1 min)")
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        device=-1,   # CPU — no CUDA required
    )

emotion_classifier = load_emotion_model()

# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def check_negation(text: str):
    """
    Returns forced_emotion string if a negation rule matches, else None.
    This runs BEFORE the transformer so it can override incorrect classifications.
    """
    lowered = text.lower()
    for pattern, forced_emotion in NEGATION_RULES:
        if re.search(pattern, lowered):
            return forced_emotion
    return None


def keyword_emotion(text: str) -> str:
    """Fallback: scan keywords to guess emotion when transformer is uncertain."""
    lowered = text.lower()
    scores = {e: 0 for e in KEYWORD_EMOTION_MAP}
    for emotion, keywords in KEYWORD_EMOTION_MAP.items():
        for kw in keywords:
            if kw in lowered:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def detect_emotion(text: str) -> tuple:
    """
    Returns (emotion, confidence, method).

    Priority order:
      1. Negation rules  → override transformer if negation pattern found
      2. Transformer     → use if confidence >= threshold
      3. Keyword fallback → use if transformer confidence too low
    """
    # ── Stage 1: Negation check ──
    negation_override = check_negation(text)
    if negation_override:
        # Still run transformer for confidence display, but override label
        results = emotion_classifier(text)[0]
        results.sort(key=lambda x: x["score"], reverse=True)
        conf = results[0]["score"]
        return negation_override, conf, "negation-override"

    # ── Stage 2: Transformer ──
    results = emotion_classifier(text)[0]
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[0]

    if top["score"] >= CONFIDENCE_THRESHOLD:
        return top["label"], top["score"], "transformer"

    # ── Stage 3: Keyword fallback ──
    fallback = keyword_emotion(text)
    return fallback, top["score"], "keyword"


def extract_snippet(text: str, max_words: int = 8) -> str:
    """
    Extract a short meaningful phrase from user message.
    Strips leading filler words but preserves negations like 'not feeling well'.
    """
    # Remove leading subject pronouns only (not negations)
    leading_fillers = r"^(i\s+am\s+|i'm\s+|im\s+|i\s+feel\s+|i\s+just\s+|i\s+keep\s+)"
    cleaned = re.sub(leading_fillers, "", text.strip().lower())
    words = cleaned.split()
    snippet = " ".join(words[:max_words])
    if not snippet:
        snippet = text.strip()[:50]
    return snippet.rstrip(".,!?")


def is_crisis(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in CRISIS_KEYWORDS)


def generate_response(user_text: str, emotion: str) -> str:
    snippet = extract_snippet(user_text)
    templates = RESPONSES.get(emotion, RESPONSES["neutral"])
    template = random.choice(templates)
    return template.replace("{snippet}", snippet)

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Features")
    st.write("✅ Transformer-based Emotion Detection")
    st.write("✅ Negation Override Layer")
    st.write("✅ Keyword Fallback (when confidence low)")
    st.write("✅ Confidence Thresholding (≥50%)")
    st.write("✅ Contextual CBT Responses")
    st.write("✅ Multi-turn Chat")
    st.write("✅ Crisis Safety Detection")
    st.write("✅ Emotion Probability Display")
    st.write("🚧 RAG Integration")
    st.write("🚧 Long-Term Memory")
    st.write("🚧 Fine-tuned Therapeutic LLM")
    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "⚠️ This is an academic NLP prototype and not a replacement "
        "for professional mental health care."
    )

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────

st.title("🧠 EmpathAI — Mental Health Support")
st.caption("Prototype v3.2 · Transformer + negation override + keyword fallback · CBT-informed responses")

st.warning(
    "⚠️ **Disclaimer:** This is a research prototype. It is NOT a licensed therapy service. "
    "Do not rely on it for clinical decisions.",
    icon="⚠️"
)

# ──────────────────────────────────────────────
#  DISPLAY HISTORY
# ──────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "user" and "emotion" in msg:
            emotion = msg["emotion"]
            conf    = msg["confidence"]
            method  = msg.get("method", "transformer")
            color   = EMOTION_COLORS.get(emotion, "#7f8c8d")
            st.markdown(
                f'<span class="emotion-pill" style="background:{color}22;'
                f'color:{color};border:1px solid {color}55">'
                f'{emotion} · {conf:.0%} confidence · [{method}]</span>',
                unsafe_allow_html=True,
            )
            st.progress(int(conf * 100))

# ──────────────────────────────────────────────
#  CHAT INPUT
# ──────────────────────────────────────────────

user_input = st.chat_input("How are you feeling today?")

if user_input:

    # ── Crisis check FIRST ─────────────────────
    if is_crisis(user_input):
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(
                '<div class="crisis-box">' + CRISIS_RESPONSE + '</div>',
                unsafe_allow_html=True,
            )
        st.session_state.messages.append({"role": "user",      "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": CRISIS_RESPONSE})
        st.stop()

    # ── Emotion detection ──────────────────────
    emotion, confidence, method = detect_emotion(user_input)

    # ── Display user message ───────────────────
    with st.chat_message("user"):
        st.markdown(user_input)
        color = EMOTION_COLORS.get(emotion, "#7f8c8d")
        st.markdown(
            f'<span class="emotion-pill" style="background:{color}22;'
            f'color:{color};border:1px solid {color}55">'
            f'{emotion} · {confidence:.0%} confidence · [{method}]</span>',
            unsafe_allow_html=True,
        )
        st.progress(int(confidence * 100))

    # ── Store user message ─────────────────────
    st.session_state.messages.append({
        "role":       "user",
        "content":    user_input,
        "emotion":    emotion,
        "confidence": confidence,
        "method":     method,
    })

    # ── Generate & display response ────────────
    reply = generate_response(user_input, emotion)
    with st.chat_message("assistant"):
        st.markdown(reply)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": reply,
    })
