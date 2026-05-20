import streamlit as st
from transformers import pipeline
import random

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

CONFIDENCE_THRESHOLD = 0.45

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

KEYWORD_EMOTION_MAP = {
    "sadness": [
        "sad", "depressed", "unhappy", "miserable", "hopeless", "lonely",
        "grief", "crying", "heartbroken", "empty", "worthless", "numb",
        "meaningless", "nothing feels", "can't feel", "don't feel",
        "lost interest", "no motivation", "tired of", "exhausted",
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
        "positive", "good day", "feeling good",
    ],
    "disgust": [
        "disgusted", "gross", "horrible", "repulsed", "revolting",
    ],
}

RESPONSES = {
    "sadness": [
        "It sounds like you're carrying something heavy right now — the feeling that '{snippet}' can be really exhausting. That kind of emotional weight is valid, and it doesn't mean things will always feel this way. What do you think has been contributing most to this feeling lately?",
        "When you say '{snippet}', I hear real pain in that. Sadness like this often comes in waves, and it's okay to let yourself feel it rather than push it away. Is there one small thing today that gave you even the tiniest sense of comfort?",
        "'{snippet}' — that's a deeply human experience, even when it doesn't feel that way. In CBT, we often look at whether our thoughts are telling us the full story. Does this feeling connect to a specific situation, or has it just been a general weight?",
    ],
    "fear": [
        "Imagining '{snippet}' puts your mind in a state of high alert — that's your brain trying to protect you, even if it's causing more harm than good. A CBT technique that helps: ask yourself — what's the realistic probability of the worst case actually happening? What evidence supports it?",
        "When our thoughts fixate on '{snippet}', it's often a sign of catastrophizing — where the mind fast-forwards to the worst outcome. Try grounding yourself: name 5 things you can see right now. Then let's talk about what's actually in your control here.",
        "'{snippet}' — that pattern of worry is exhausting. One CBT tool is scheduling a dedicated 'worry window' of 10 minutes per day where you allow yourself to think about fears, then consciously set them aside. Have you tried anything like that before?",
    ],
    "anger": [
        "Feeling like '{snippet}' makes complete sense — when we feel unheard or disrespected repeatedly, anger is a natural response. Before acting on it, can you identify the core need underneath the anger? Often it's a need for respect, fairness, or control.",
        "'{snippet}' — that frustration is real. Anger often sits on top of a deeper emotion like hurt or helplessness. What do you think is underneath it in this case?",
        "When '{snippet}' keeps happening, it can feel like you have no control. A CBT approach here is to separate what you *can* control from what you can't — and focus your energy only on the former. What's one part of this situation you actually have influence over?",
    ],
    "joy": [
        "I'm really glad to hear '{snippet}'! Positive moments are worth pausing on — our brains have a negativity bias, so consciously savouring good feelings actually rewires neural patterns over time. What made this moment feel good?",
        "'{snippet}' — that's genuinely wonderful. How are you feeling about things overall right now compared to earlier?",
    ],
    "disgust": [
        "It sounds like '{snippet}' has left you feeling really uncomfortable. That reaction often signals that something violated a value you hold. What specifically felt wrong about it?",
    ],
    "surprise": [
        "'{snippet}' — sounds like something unexpected happened. Are you feeling more positive or unsettled about it?",
    ],
    "neutral": [
        "You mentioned '{snippet}'. I want to make sure I understand what you're going through — can you tell me a bit more about how that's been making you feel?",
        "Thank you for sharing that. When you say '{snippet}', what emotion comes up most strongly for you right now?",
        "I hear you. '{snippet}' — sometimes it's hard to name what we're feeling. Would you describe it more as heaviness, tension, numbness, or something else?",
    ],
}

# ──────────────────────────────────────────────
#  LOAD MODEL  — uses HuggingFace hub cache
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading emotion model… (first run takes ~1 min)")
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None,
        # explicitly use CPU so no CUDA dependency needed on cloud
        device=-1,
    )

emotion_classifier = load_emotion_model()

# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def keyword_emotion(text: str) -> str:
    lowered = text.lower()
    scores = {e: 0 for e in KEYWORD_EMOTION_MAP}
    for emotion, keywords in KEYWORD_EMOTION_MAP.items():
        for kw in keywords:
            if kw in lowered:
                scores[emotion] += 1
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def detect_emotion(text: str) -> tuple:
    results = emotion_classifier(text)[0]
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[0]
    if top["score"] >= CONFIDENCE_THRESHOLD:
        return top["label"], top["score"], "transformer"
    fallback = keyword_emotion(text)
    return fallback, top["score"], "keyword"


def extract_snippet(text: str, max_words: int = 7) -> str:
    fillers = {"i", "i'm", "im", "i've", "ive", "i feel", "i am", "i keep", "i just"}
    words = text.strip().lower().split()
    while words and words[0] in fillers:
        words.pop(0)
    snippet = " ".join(words[:max_words])
    if not snippet:
        snippet = text.strip()[:40]
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
    st.write("✅ Keyword Fallback (when confidence low)")
    st.write("✅ Confidence Thresholding (≥45%)")
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
st.caption("Prototype v3.1 · Transformer + keyword emotion detection · CBT-informed responses")

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
