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

st.title("🧠 Transformer-Based NLP Prototype")

st.caption(
    "AI-powered CBT-inspired mental health support chatbot "
    "using Transformer NLP, semantic emotion detection, "
    "and contextual empathetic reframing."
)

# ──────────────────────────────────────────────
# CASUAL / SMART INTENTS
# ──────────────────────────────────────────────

CASUAL_INTENTS = {

    "greeting": {
        "patterns": [
            "hello", "hi", "hey",
            "good morning",
            "good evening",
            "yo"
        ],
        "responses": [
            "Hello 👋 I'm here to listen. What's been on your mind recently?",
            "Hey — I'm glad you reached out today. How have things been feeling lately?",
            "Hi there. How are you feeling emotionally today?"
        ]
    },

    "how_are_you": {
        "patterns": [
            "how are you",
            "how are u",
            "how's it going"
        ],
        "responses": [
            "I'm here and ready to support you. More importantly — how have you been feeling lately?",
            "I'm doing well and focused on helping you. What's been on your mind today?"
        ]
    },

    "thanks": {
        "patterns": [
            "thank you",
            "thanks",
            "thx"
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
            "Take care of yourself. Remember — difficult emotions are temporary.",
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
            "You do not need to solve your entire life today. Small consistent steps matter.",
            "Progress usually feels slow while it's happening. That doesn't mean you're failing."
        ]
    },

    "lonely": {
        "patterns": [
            "i feel lonely",
            "i feel alone",
            "nobody understands me"
        ],
        "responses": [
            "Feeling emotionally disconnected from people can be deeply painful.",
            "Loneliness can distort how we see ourselves and others."
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
            "Relationship pain affects emotional stability because attachment is deeply human.",
            "Emotional loss often causes repetitive thoughts and overanalysis."
        ]
    },

    "study_stress": {
        "patterns": [
            "exam stress",
            "placement pressure",
            "study pressure",
            "career tension"
        ],
        "responses": [
            "Academic pressure can make it feel like your future depends on one outcome.",
            "Stress narrows thinking and increases fear-based thoughts."
        ]
    }
}

# ──────────────────────────────────────────────
# INTENT DETECTION
# ──────────────────────────────────────────────

def detect_intent(text):

    lowered = text.lower()

    for intent, data in CASUAL_INTENTS.items():

        for pattern in data["patterns"]:

            if pattern in lowered:
                return intent

    return None

# ──────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────

@st.cache_resource
def load_model():

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

classifier = load_model()

# ──────────────────────────────────────────────
# EMOTION DETECTION
# ──────────────────────────────────────────────

def detect_emotion(user_input):

    results = classifier(user_input)[0]

    top_result = max(results, key=lambda x: x["score"])

    emotion = top_result["label"].lower()
    confidence = round(top_result["score"] * 100)

    text = user_input.lower()

    # ──────────────────────────────────────────
    # SEMANTIC NLP CORRECTION LAYER
    # ──────────────────────────────────────────

    sad_words = [
        "sad", "hopeless", "empty", "lonely",
        "worthless", "tired", "drained",
        "crying", "depressed", "broken",
        "hurt", "numb", "exhausted",
        "meaningless", "upset", "disconnected"
    ]

    fear_words = [
        "anxious", "anxiety", "fear",
        "worried", "panic", "overthinking",
        "scared", "future", "stress",
        "pressure", "nervous",
        "racing thoughts", "placement",
        "exam"
    ]

    anger_words = [
        "angry", "frustrated",
        "annoyed", "irritated",
        "mad", "hate", "furious"
    ]

    joy_words = [
        "happy", "peaceful",
        "excited", "good",
        "better", "great",
        "fine", "relaxed",
        "motivated", "proud"
    ]

    negative_context = [
        "not", "never", "don't",
        "cant", "can't", "hardly"
    ]

    # ── sadness detection ──

    if (
        any(word in text for word in sad_words)
        or (
            any(n in text for n in negative_context)
            and any(j in text for j in joy_words)
        )
    ):

        emotion = "sadness"
        confidence = 95
        method = "semantic NLP override"

    # ── fear detection ──

    elif any(word in text for word in fear_words):

        emotion = "fear"
        confidence = 92
        method = "semantic NLP override"

    # ── anger detection ──

    elif any(word in text for word in anger_words):

        emotion = "anger"
        confidence = 89
        method = "semantic NLP override"

    # ── joy detection ──

    elif any(word in text for word in joy_words):

        emotion = "joy"
        confidence = 90
        method = "semantic NLP override"

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
# DISPLAY CHAT HISTORY
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
            "I'm really sorry you're feeling this overwhelmed right now.\n\n"
            "You deserve immediate support from trusted people or professionals. "
            "Please consider reaching out to a mental health professional, "
            "trusted friend, or crisis helpline."
        )

    elif emotion == "sadness":

        bot_reply = (
            "It sounds like emotional exhaustion and painful thoughts "
            "have been weighing heavily on you.\n\n"
            "Sometimes emotions convince us that pain will last forever, "
            "even though emotional states naturally change over time.\n\n"
            "Taking one small meaningful step today may slowly help rebuild emotional balance."
        )

    elif emotion == "fear":

        bot_reply = (
            "Your mind seems focused on uncertainty and future possibilities right now.\n\n"
            "Anxiety often pushes the brain into worst-case prediction mode. "
            "Grounding yourself in the present moment can reduce mental overload."
        )

    elif emotion == "anger":

        bot_reply = (
            "It seems like emotional pressure and frustration have been building internally.\n\n"
            "Strong emotions can speed up thoughts and reactions. "
            "Pausing before reacting may help you regain clarity and emotional control."
        )

    elif emotion == "joy":

        bot_reply = (
            "I'm glad something positive is happening emotionally for you right now.\n\n"
            "The brain naturally focuses more on negative experiences, "
            "so intentionally noticing positive moments strengthens emotional resilience."
        )

    else:

        bot_reply = (
            "Thank you for sharing your thoughts openly.\n\n"
            "Talking about emotions instead of suppressing them "
            "can reduce emotional pressure and improve self-awareness."
        )

    # ── Store Assistant Message ──────────────

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()
