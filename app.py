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
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
.main {
    background-color: #020617;
}
.stChatMessage {
    border-radius: 14px;
    padding: 10px;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
st.title("🧠 EmpathAI")
st.caption(
    "Transformer-based NLP Prototype for Mental Health Support, "
    "CBT Reframing, Semantic Emotion Detection, and Crisis Safety Monitoring."
)
st.warning(
    "⚠️ Academic NLP Prototype — Not a replacement for professional mental health care."
)

# ──────────────────────────────────────────────
# CASUAL / SMART INTENTS
# ──────────────────────────────────────────────
CASUAL_INTENTS = {
    "greeting": {
        "patterns": [
            "hello", "hi", "hey",
            "good morning",
            "good evening"
        ],
        "responses": [
            "Hello 👋 I'm here to listen. What's been on your mind recently?",
            "Hey — I'm glad you reached out today. How have things been feeling lately?",
            "Hi there. How are you feeling emotionally today?"
        ]
    },
    "thanks": {
        "patterns": [
            "thank you",
            "thanks",
            "thx"
        ],
        "responses": [
            "You're welcome. Talking openly already takes courage.",
            "I'm glad you shared your thoughts with me."
        ]
    },
    "bye": {
        "patterns": [
            "bye",
            "goodbye",
            "take care"
        ],
        "responses": [
            "Take care of yourself. Difficult emotions are temporary.",
            "I'm glad we talked today. Be gentle with yourself."
        ]
    },
    "motivation": {
        "patterns": [
            "motivate me",
            "i need motivation"
        ],
        "responses": [
            "You do not need to solve your entire life today. Small consistent steps matter.",
            "Progress feels slow while it's happening. That doesn't mean you're failing."
        ]
    }
}

# ──────────────────────────────────────────────
# PROACTIVE FOLLOW-UP QUESTIONS
# ──────────────────────────────────────────────
FOLLOWUPS = {
    "sadness": [
        "\n\nCan you tell me more — when did this feeling start?",
        "\n\nIs there something specific that's been weighing on you?",
        "\n\nHave you been able to talk to anyone about this?"
    ],
    "fear": [
        "\n\nWhat feels most overwhelming right now?",
        "\n\nIs there one specific thing driving this anxiety?",
        "\n\nHow long have you been carrying this worry?"
    ],
    "anger": [
        "\n\nWhat triggered this feeling today?",
        "\n\nDo you feel like this happens often, or is today different?",
        "\n\nIs there someone or something specific making you feel this way?"
    ],
    "joy": [
        "\n\nWhat's been making you feel this way? I'd love to hear more.",
        "\n\nThat's great to hear — what's been going well for you?",
        "\n\nHave you been able to share this with anyone close to you?"
    ],
    "neutral": [
        "\n\nWhat's been on your mind lately?",
        "\n\nIs there something specific you'd like to talk through today?",
        "\n\nHow have you been sleeping and taking care of yourself recently?"
    ],
    "disgust": [
        "\n\nWhat's been making you feel this way?",
        "\n\nIs this something that happened recently or has it been building up?"
    ],
    "surprise": [
        "\n\nThat sounds unexpected — do you want to talk about what happened?",
        "\n\nHow are you processing that? Are you okay?"
    ]
}

# ──────────────────────────────────────────────
# CHECKIN PROMPTS (every 4 turns)
# ──────────────────────────────────────────────
CHECKIN_PROMPTS = [
    "\n\n💬 *Check-in: How are you feeling compared to when we started talking?*",
    "\n\n💬 *It's been a little while — are you doing okay? Is there anything else on your mind?*",
    "\n\n💬 *Sometimes just talking helps. How is your emotional energy right now?*"
]

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
# LOAD TRANSFORMER MODEL
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

    sad_words = [
        "sad", "hopeless", "empty",
        "worthless", "lonely",
        "crying", "depressed",
        "broken", "hurt",
        "meaningless", "tired",
        "bad", "not okay",
        "not fine", "upset"
    ]
    fear_words = [
        "anxious", "stress",
        "panic", "future",
        "worried", "overthinking",
        "pressure", "exam",
        "placement", "fear"
    ]
    anger_words = [
        "angry", "frustrated",
        "hate", "annoyed",
        "irritated"
    ]
    joy_words = [
        "happy", "peaceful",
        "great", "better",
        "motivated", "proud",
        "relaxed", "excited"
    ]
    negative_phrases = [
        "not feeling good",
        "not good",
        "dont feel good",
        "don't feel good",
        "not feeling well",
        "feeling bad",
        "feeling terrible",
        "i am not okay",
        "i feel horrible"
    ]

    if any(phrase in text for phrase in negative_phrases):
        emotion = "sadness"
        confidence = 94
        method = "semantic NLP override"
    elif any(word in text for word in sad_words):
        emotion = "sadness"
        confidence = 95
        method = "semantic NLP override"
    elif any(word in text for word in fear_words):
        emotion = "fear"
        confidence = 92
        method = "semantic NLP override"
    elif any(word in text for word in anger_words):
        emotion = "anger"
        confidence = 89
        method = "semantic NLP override"
    elif any(word in text for word in joy_words):
        emotion = "joy"
        confidence = 90
        method = "semantic NLP override"
    else:
        method = "transformer"

    return emotion, confidence, method

# ──────────────────────────────────────────────
# SESSION STATE INIT
# ──────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.turn_count = 0
    # ── PROACTIVE OPENING MESSAGE ──
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hey 👋 I'm EmpathAI — I'm here to listen, no judgment, no rush.\n\n"
            "How has your day been feeling so far? You can share as little or as much as you'd like."
        )
    })

if "turn_count" not in st.session_state:
    st.session_state.turn_count = 0

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Features")
    st.write("✅ Transformer-based Emotion Detection")
    st.write("✅ Semantic NLP Correction")
    st.write("✅ CBT-inspired Responses")
    st.write("✅ Intent Detection Layer")
    st.write("✅ Multi-turn Chat")
    st.write("✅ Crisis Safety Detection")
    st.write("✅ Proactive Opening Message")
    st.write("✅ Follow-up Questions")
    st.write("✅ Periodic Check-ins")
    st.write("🚧 RAG Integration")
    st.write("🚧 Long-Term Memory")
    st.write("🚧 Personalized Support")
    st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.turn_count = 0
        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                "Hey 👋 I'm EmpathAI — I'm here to listen, no judgment, no rush.\n\n"
                "How has your day been feeling so far? You can share as little or as much as you'd like."
            )
        })
        st.rerun()
    st.divider()
    st.info("Prototype for NLP-based Mental Health Support System")

# ──────────────────────────────────────────────
# DISPLAY CHAT HISTORY
# ──────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "emotion" in message:
            st.caption(
                f"Emotion: {message['emotion']} • "
                f"{message['confidence']}% confidence • "
                f"[{message['method']}]"
            )
            st.progress(message["confidence"] / 100)

# ──────────────────────────────────────────────
# CHAT INPUT
# ──────────────────────────────────────────────
user_input = st.chat_input("How are you feeling today?")

if user_input:

    # ── INTENT DETECTION ──
    intent = detect_intent(user_input)
    if intent:
        reply = random.choice(CASUAL_INTENTS[intent]["responses"])
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── EMOTION DETECTION ──
    emotion, confidence, method = detect_emotion(user_input)
    text = user_input.lower()

    # ── CRISIS DETECTION ──
    crisis_keywords = [
        "suicide",
        "kill myself",
        "end my life",
        "don't want to live",
        "want to disappear",
        "self harm",
        "hurt myself",
        "i want to die"
    ]
    crisis_detected = any(word in text for word in crisis_keywords)

    # ── STORE USER MESSAGE ──
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "emotion": emotion,
        "confidence": confidence,
        "method": method
    })

    # ── INCREMENT TURN COUNT ──
    st.session_state.turn_count += 1

    # ── GENERATE RESPONSE ──
    if crisis_detected:
        bot_reply = (
            "I'm really sorry you're feeling this overwhelmed right now.\n\n"
            "🇮🇳 India Mental Health Helplines:\n"
            "📞 Tele-MANAS: 14416 or 1-800-891-4416\n"
            "📞 AASRA Suicide Prevention: +91 9820466726\n\n"
            "Please reach out to a trusted person or mental health professional immediately.\n\n"
            "You are not alone."
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

    # ── PROACTIVE FOLLOW-UP (not for crisis) ──
    if not crisis_detected:
        followup_list = FOLLOWUPS.get(emotion, FOLLOWUPS["neutral"])
        bot_reply += random.choice(followup_list)

    # ── PERIODIC CHECK-IN (every 4 turns, not for crisis) ──
    if not crisis_detected and st.session_state.turn_count % 4 == 0:
        bot_reply += random.choice(CHECKIN_PROMPTS)

    # ── STORE ASSISTANT MESSAGE ──
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()