import streamlit as st
from transformers import pipeline
import random

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="EmpathAI - Mental Health Support",
    page_icon="🧠",
    layout="wide"
)

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────

st.title("🧠 EmpathAI - Mental Health Support")
st.subheader("Transformer-Based NLP Prototype")

st.caption(
    "AI-powered CBT-inspired mental health support chatbot "
    "using Transformer NLP, semantic emotion detection, "
    "and contextual empathetic reframing."
)

# ──────────────────────────────────────────────
# CASUAL / SMART INTENT DETECTION
# ──────────────────────────────────────────────

CASUAL_INTENTS = {

    "greeting": {
        "patterns": [
            "hello", "hi", "hey",
            "good morning",
            "good evening",
            "good afternoon",
            "yo"
        ],
        "responses": [
            "Hey — I'm glad you reached out today. How have things been feeling lately?",
            "Hello 👋 I'm here to listen. What's been on your mind recently?",
            "Hi there. How are you feeling emotionally today?"
        ]
    },

    "how_are_you": {
        "patterns": [
            "how are you",
            "how's it going",
            "how are u"
        ],
        "responses": [
            "I'm here and ready to support you. More importantly — how have *you* been feeling lately?",
            "I'm doing well and focused on helping you. What's been on your mind today?"
        ]
    },

    "thanks": {
        "patterns": [
            "thank you",
            "thanks",
            "thx",
            "appreciate it"
        ],
        "responses": [
            "You're always welcome. I'm glad you shared your thoughts with me.",
            "No need to thank me — reaching out and talking openly already takes courage."
        ]
    },

    "bye": {
        "patterns": [
            "bye",
            "goodbye",
            "see you",
            "take care"
        ],
        "responses": [
            "Take care of yourself. Remember — difficult emotions are temporary, even when they feel overwhelming.",
            "I'm glad we talked today. Be gentle with yourself."
        ]
    },

    "motivation": {
        "patterns": [
            "motivate me",
            "i need motivation",
            "give me motivation"
        ],
        "responses": [
            "You do not need to solve your entire life today. Small consistent steps are more powerful than temporary bursts of motivation.",
            "Progress usually looks slow while it's happening. That doesn't mean you're failing."
        ]
    },

    "lonely": {
        "patterns": [
            "i feel lonely",
            "nobody understands me",
            "i feel alone"
        ],
        "responses": [
            "Feeling emotionally disconnected from people can be deeply painful. But being alone right now does not mean you'll always feel this way.",
            "Loneliness often convinces us that nobody cares — even when that isn't fully true. Have you been isolating yourself lately?"
        ]
    },

    "relationship": {
        "patterns": [
            "breakup",
            "she left me",
            "he left me",
            "relationship problem"
        ],
        "responses": [
            "Relationship pain can affect every part of daily life because emotional attachment is deeply wired into the brain.",
            "Right now your mind may be replaying painful thoughts repeatedly. That's normal after emotional loss."
        ]
    },

    "study_stress": {
        "patterns": [
            "exam stress",
            "placement pressure",
            "career tension",
            "study pressure"
        ],
        "responses": [
            "Academic pressure can make it feel like your entire future depends on one outcome — but your journey is much larger than a single exam or placement.",
            "Stress narrows attention and increases fear-based thinking. Try focusing only on the next manageable step."
        ]
    }
}

# ──────────────────────────────────────────────
# INTENT DETECTION FUNCTION
# ──────────────────────────────────────────────

def detect_intent(text):

    lowered = text.lower()

    for intent, data in CASUAL_INTENTS.items():

        for pattern in data["patterns"]:

            if pattern in lowered:
                return intent

    return None

# ──────────────────────────────────────────────
# LOAD TRANSFORMER MODEL
# ──────────────────────────────────────────────

@st.cache_resource
def load_emotion_model():

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

classifier = load_emotion_model()

# ──────────────────────────────────────────────
# EMOTION DETECTION
# ──────────────────────────────────────────────

def detect_emotion(user_input):

    results = classifier(user_input)[0]

    top_result = max(results, key=lambda x: x["score"])

    emotion = top_result["label"].lower()
    confidence = round(top_result["score"] * 100)

    text = user_input.lower()

    # ── semantic correction layer ──

    sad_patterns = [
        "not feeling well",
        "nothing feels meaningful",
        "emotionally numb",
        "empty inside",
        "nothing matters",
        "feel drained",
        "lost interest",
        "i feel empty",
        "i feel hopeless",
        "feeling terrible",
        "feeling awful",
        "feeling bad",
        "mentally exhausted",
        "emotionally tired"
    ]

    fear_patterns = [
        "worst-case",
        "overthinking",
        "thoughts racing",
        "can't stop thinking",
        "future scares me",
        "keep imagining",
        "mind won't stop",
        "my thoughts won't stop"
    ]

    anger_patterns = [
        "everyone irritates me",
        "frustrated",
        "annoyed",
        "angry lately",
        "people keep irritating me"
    ]

    joy_patterns = [
        "feeling better",
        "feeling happy",
        "peaceful day",
        "feeling good today",
        "i feel okay now"
    ]

    # ── semantic overrides ──

    if any(p in text for p in sad_patterns):
        emotion = "sadness"
        confidence = 95
        method = "semantic override"

    elif any(p in text for p in fear_patterns):
        emotion = "fear"
        confidence = 92
        method = "semantic override"

    elif any(p in text for p in anger_patterns):
        emotion = "anger"
        confidence = 89
        method = "semantic override"

    elif any(p in text for p in joy_patterns):
        emotion = "joy"
        confidence = 90
        method = "semantic override"

    else:
        method = "transformer"

    return emotion, confidence, method

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────

with st.sidebar:

    st.header("⚙️ Features")

    st.write("✅ Transformer-based Emotion Detection")
    st.write("✅ Semantic Emotion Correction")
    st.write("✅ CBT-inspired Responses")
    st.write("✅ Intent Detection Layer")
    st.write("✅ Multi-turn Chat")
    st.write("✅ Crisis Safety Detection")

    st.write("🚧 RAG Integration")
    st.write("🚧 Long-Term Memory")
    st.write("🚧 Personalized Therapy Support")

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.warning(
        "⚠️ This is an academic NLP prototype and "
        "not a replacement for professional mental health care."
    )

# ──────────────────────────────────────────────
# DISPLAY OLD CHAT
# ──────────────────────────────────────────────

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if "emotion" in message:

            st.caption(
                f"{message['emotion']} • "
                f"{message['confidence']}% confidence "
                f"• [{message['method']}]"
            )

            st.progress(message["confidence"] / 100)

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────

user_input = st.chat_input("How are you feeling today?")

if user_input:

    # ── Intent Detection ─────────────────────

    intent = detect_intent(user_input)

    if intent:

        reply = random.choice(
            CASUAL_INTENTS[intent]["responses"]
        )

        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": reply
        })

        st.rerun()

    # ── Emotion Detection ────────────────────

    emotion, confidence, method = detect_emotion(user_input)

    text = user_input.lower()

    # ── Crisis Detection ─────────────────────

    crisis_keywords = [
        "suicide",
        "kill myself",
        "end my life",
        "don't want to live",
        "want to disappear",
        "self harm"
    ]

    crisis_detected = any(
        word in text for word in crisis_keywords
    )

    # ── Store User Message ───────────────────

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "emotion": emotion,
        "confidence": confidence,
        "method": method
    })

    # ── Generate Response ────────────────────

    if crisis_detected:

        bot_reply = (
            "I'm really sorry you're feeling this overwhelmed right now. "
            "You deserve support from real people who can help immediately.\n\n"
            "Please consider reaching out to a trusted person, mental health "
            "professional, or local crisis helpline."
        )

    elif emotion == "sadness":

        bot_reply = (
            "It sounds like emotional exhaustion and hopeless thoughts "
            "have been weighing heavily on you.\n\n"
            "Sometimes the brain starts treating painful emotions as permanent truths — "
            "even though emotions naturally change over time.\n\n"
            "Focusing on one small meaningful step today may help create emotional momentum again."
        )

    elif emotion == "fear":

        bot_reply = (
            "Your mind seems highly focused on uncertainty and possible future outcomes right now.\n\n"
            "Anxiety often pushes the brain into worst-case prediction mode. "
            "Grounding yourself in the present moment instead of imagined futures "
            "can slowly reduce mental overload."
        )

    elif emotion == "anger":

        bot_reply = (
            "It seems like frustration and emotional pressure have been building internally for some time.\n\n"
            "Strong emotions can speed up thoughts and reactions. "
            "Pausing briefly before reacting may help you regain clarity and emotional control."
        )

    elif emotion == "joy":

        bot_reply = (
            "I'm glad something positive is happening emotionally for you right now.\n\n"
            "The brain naturally focuses more on negative experiences, "
            "so intentionally noticing positive moments can strengthen emotional resilience over time."
        )

    else:

        bot_reply = (
            "Thank you for sharing your thoughts openly.\n\n"
            "Talking about emotions instead of suppressing them "
            "can sometimes reduce emotional pressure and increase self-awareness."
        )

    # ── Store Assistant Message ──────────────

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()
