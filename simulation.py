import streamlit as st
from transformers import pipeline
import random
import csv
import os
import uuid
import time
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="EmpathAI — Agent Simulation",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 EmpathAI — Agent Simulation")
st.caption("Patient Bot vs Therapist Bot — Simulated Counseling Session")
st.info(
    "🔬 This simulation is developed solely for **academic and research purposes**. "
    "It is NOT intended for clinical application."
)

# ──────────────────────────────────────────────
# LOAD EMOTION MODEL
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
# PATIENT SCENARIOS
# ──────────────────────────────────────────────
PATIENT_SCENARIOS = {
    "Anxiety & Overthinking": {
        "intro": "I've been feeling really overwhelmed lately. I can't stop overthinking everything.",
        "messages": [
            "I keep worrying about things that haven't even happened yet.",
            "My mind races at night and I can't sleep properly.",
            "I feel like something bad is always about to happen.",
            "Even small decisions feel really stressful for me.",
            "I get panic attacks sometimes when I'm under pressure.",
            "I'm scared about my future and I don't know what to do.",
            "I feel like I'm always on edge, even when nothing is wrong.",
            "Sometimes I just want to shut everything out and be alone."
        ]
    },
    "Depression & Sadness": {
        "intro": "I've been feeling really empty and low for a while now.",
        "messages": [
            "I don't feel motivated to do anything anymore.",
            "I used to enjoy things but now nothing feels good.",
            "I feel worthless most of the time.",
            "Getting out of bed feels like a huge task every morning.",
            "I feel like a burden to everyone around me.",
            "I've been crying a lot for no specific reason.",
            "I don't see the point in trying anymore.",
            "I feel completely disconnected from people I used to be close to."
        ]
    },
    "Anger & Frustration": {
        "intro": "I'm really frustrated and I feel like nobody understands me.",
        "messages": [
            "I get angry very quickly and I don't know how to control it.",
            "People keep letting me down and it makes me furious.",
            "I said things I regret when I was angry last week.",
            "I feel like everyone is against me.",
            "Small things set me off and I can't stop it.",
            "I punched a wall last month because I was so angry.",
            "I feel like I'm always being disrespected.",
            "My anger is affecting my relationships badly."
        ]
    },
    "Grief & Loss": {
        "intro": "I lost someone very close to me recently and I don't know how to cope.",
        "messages": [
            "I think about them every single day.",
            "I feel guilty like I could have done more.",
            "The house feels so empty without them.",
            "I don't know how to move forward with my life.",
            "Sometimes I forget they're gone and then it hits me all over again.",
            "I've stopped talking to friends because I don't want to be a burden.",
            "I don't know if this pain will ever go away.",
            "I'm just going through the motions every day, not really living."
        ]
    }
}

# ──────────────────────────────────────────────
# THERAPIST RESPONSES (emotion-based CBT)
# ──────────────────────────────────────────────
THERAPIST_RESPONSES = {
    "sadness": [
        "It sounds like you're carrying a lot of emotional weight right now. That takes real courage to acknowledge. What do you think has been the hardest part of this for you?",
        "Depression can make everything feel heavier and more hopeless than it actually is. Have you been able to identify any small moments of relief, even briefly?",
        "What you're feeling is valid. Sadness doesn't mean weakness — it means you care deeply. Can you tell me more about when this feeling is strongest?",
        "Sometimes our mind tells us we're a burden, but that's the depression talking, not the truth. Who in your life do you feel most comfortable being honest with?"
    ],
    "fear": [
        "Anxiety thrives on uncertainty. Let's try to separate what's in your control from what isn't — that often helps reduce the mental overload. What feels most urgent to you right now?",
        "That racing mind at night is very common with anxiety. Have you tried any grounding techniques before sleep, like focusing on your breathing?",
        "Worry about the future is one of anxiety's biggest tools. What would you say to a close friend who was feeling exactly what you're feeling?",
        "Panic attacks are frightening but they always pass. Your body is trying to protect you, even when there's no real threat. What usually triggers yours?"
    ],
    "anger": [
        "Anger is often a secondary emotion — it protects us from something deeper like hurt or fear. What do you think is underneath the anger for you?",
        "It takes a lot of self-awareness to recognize that anger is affecting your relationships. What does it feel like in your body right before you lose your temper?",
        "Feeling disrespected repeatedly is genuinely painful. Have you been able to communicate your boundaries clearly to the people involved?",
        "Anger is energy — it needs somewhere to go. Have you found any physical outlets like exercise that help discharge it before it builds up?"
    ],
    "joy": [
        "I'm glad there's something positive here. Even in difficult times, noticing what feels okay is important. Can you tell me more about that?",
        "That's a good sign. What do you think is helping you feel this way, even slightly?",
    ],
    "neutral": [
        "Thank you for sharing that with me. Can you help me understand a bit more about what your day-to-day has been feeling like recently?",
        "I hear you. Sometimes it's hard to put feelings into words. Is there one specific thing that's been bothering you the most?",
        "It sounds like things have been difficult. What would feel like a small step forward for you right now?",
        "You've shown a lot of courage by talking about this. What would you most like to feel differently about your situation?"
    ],
    "disgust": [
        "That sounds really uncomfortable. What specifically has been making you feel this way?",
        "Sometimes disgust signals that something has crossed a personal value for us. What feels most violated right now?"
    ],
    "surprise": [
        "Unexpected situations can really throw us off balance. How are you processing what happened?",
        "It's okay to feel unsettled by something unexpected. Give yourself time to adjust before deciding what to do next."
    ]
}

# ──────────────────────────────────────────────
# EMOTION DETECTION
# ──────────────────────────────────────────────
def detect_emotion(text):
    results = classifier(text)[0]
    top = max(results, key=lambda x: x["score"])
    emotion = top["label"].lower()
    confidence = round(top["score"] * 100)

    t = text.lower()
    sad_words = ["sad", "hopeless", "empty", "worthless", "lonely", "depressed", "broken", "hurt", "meaningless", "tired", "crying", "not okay", "upset", "low", "burden", "guilt", "empty"]
    fear_words = ["anxious", "stress", "panic", "worried", "overthinking", "fear", "scared", "overwhelmed", "racing", "edge"]
    anger_words = ["angry", "frustrated", "hate", "furious", "annoyed", "irritated", "rage", "disrespected", "punched"]
    joy_words = ["happy", "good", "better", "motivated", "proud", "relaxed", "excited", "positive"]

    if any(w in t for w in sad_words):
        return "sadness", 95
    elif any(w in t for w in fear_words):
        return "fear", 92
    elif any(w in t for w in anger_words):
        return "anger", 89
    elif any(w in t for w in joy_words):
        return "joy", 90
    return emotion, confidence

# ──────────────────────────────────────────────
# THERAPIST BOT RESPONSE
# ──────────────────────────────────────────────
def therapist_respond(patient_message):
    emotion, confidence = detect_emotion(patient_message)
    responses = THERAPIST_RESPONSES.get(emotion, THERAPIST_RESPONSES["neutral"])
    reply = random.choice(responses)
    return reply, emotion, confidence

# ──────────────────────────────────────────────
# SAVE SIMULATION LOG
# ──────────────────────────────────────────────
def save_simulation_log(session_id, turn, speaker, message, emotion="", confidence=""):
    filename = "simulation_log.csv"
    file_exists = os.path.exists(filename)
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["session_id", "timestamp", "turn", "speaker", "message", "emotion", "confidence"])
        writer.writerow([
            session_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            turn,
            speaker,
            message,
            emotion,
            confidence
        ])

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "sim_messages" not in st.session_state:
    st.session_state.sim_messages = []
if "sim_running" not in st.session_state:
    st.session_state.sim_running = False
if "sim_done" not in st.session_state:
    st.session_state.sim_done = False
if "sim_session_id" not in st.session_state:
    st.session_state.sim_session_id = str(uuid.uuid4())[:8]

# ──────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation Settings")
    scenario = st.selectbox("🧑 Patient Scenario", list(PATIENT_SCENARIOS.keys()))
    num_turns = st.slider("Number of Turns", min_value=3, max_value=8, value=5)
    st.divider()
    st.caption(f"Session ID: `{st.session_state.sim_session_id}`")
    st.divider()
    start_btn = st.button("▶️ Start Simulation", use_container_width=True)
    reset_btn = st.button("🔄 Reset", use_container_width=True)
    st.divider()
    st.write("**Legend:**")
    st.write("🧑 = Patient Bot")
    st.write("🧠 = Therapist Bot")

if reset_btn:
    st.session_state.sim_messages = []
    st.session_state.sim_running = False
    st.session_state.sim_done = False
    st.session_state.sim_session_id = str(uuid.uuid4())[:8]
    st.rerun()

# ──────────────────────────────────────────────
# DISPLAY EXISTING MESSAGES
# ──────────────────────────────────────────────
st.subheader(f"📋 Simulated Session — {scenario}")
st.divider()

chat_container = st.container()
with chat_container:
    for msg in st.session_state.sim_messages:
        if msg["speaker"] == "Patient":
            with st.chat_message("user"):
                st.markdown(f"**🧑 Patient:** {msg['message']}")
                if msg.get("emotion"):
                    st.caption(f"Detected emotion: {msg['emotion']} • {msg['confidence']}% confidence")
                    st.progress(msg["confidence"] / 100)
        else:
            with st.chat_message("assistant"):
                st.markdown(f"**🧠 Therapist:** {msg['message']}")

# ──────────────────────────────────────────────
# RUN SIMULATION
# ──────────────────────────────────────────────
if start_btn and not st.session_state.sim_done:
    st.session_state.sim_messages = []
    st.session_state.sim_running = True
    st.session_state.sim_session_id = str(uuid.uuid4())[:8]

    scenario_data = PATIENT_SCENARIOS[scenario]
    patient_messages = [scenario_data["intro"]] + scenario_data["messages"]

    progress_bar = st.progress(0)
    status = st.empty()

    for turn in range(num_turns):
        # ── PATIENT SPEAKS ──
        patient_msg = patient_messages[turn % len(patient_messages)]
        emotion, confidence = detect_emotion(patient_msg)

        status.info(f"🧑 Patient is speaking... (Turn {turn + 1}/{num_turns})")
        time.sleep(1)

        st.session_state.sim_messages.append({
            "speaker": "Patient",
            "message": patient_msg,
            "emotion": emotion,
            "confidence": confidence
        })
        save_simulation_log(
            st.session_state.sim_session_id,
            turn + 1, "Patient", patient_msg, emotion, confidence
        )

        # ── THERAPIST RESPONDS ──
        status.info(f"🧠 Therapist is responding... (Turn {turn + 1}/{num_turns})")
        time.sleep(1.5)

        therapist_reply, _, _ = therapist_respond(patient_msg)
        st.session_state.sim_messages.append({
            "speaker": "Therapist",
            "message": therapist_reply,
            "emotion": "",
            "confidence": ""
        })
        save_simulation_log(
            st.session_state.sim_session_id,
            turn + 1, "Therapist", therapist_reply
        )

        progress_bar.progress((turn + 1) / num_turns)

    status.success("✅ Simulation complete!")
    st.session_state.sim_running = False
    st.session_state.sim_done = True
    st.rerun()

# ──────────────────────────────────────────────
# DOWNLOAD BUTTON
# ──────────────────────────────────────────────
if st.session_state.sim_done and os.path.exists("simulation_log.csv"):
    st.divider()
    st.success(f"✅ Simulation complete — {len(st.session_state.sim_messages)} messages logged.")
    with open("simulation_log.csv", "rb") as f:
        st.download_button(
            label="⬇️ Download Simulation Log (CSV)",
            data=f,
            file_name=f"simulation_{st.session_state.sim_session_id}.csv",
            mime="text/csv"
        )
