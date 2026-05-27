# ──────────────────────────────────────────────
# resources.py
# Mental Health Resources, Helplines, CBT Tips,
# and Grounding Exercises for EmpathAI
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
# CRISIS HELPLINES (India)
# ──────────────────────────────────────────────
CRISIS_HELPLINES = {
    "iCall": "9152987821",
    "Vandrevala Foundation": "1860-2662-345",
    "AASRA Suicide Prevention": "+91 9820466627",
    "Tele-MANAS (Govt. of India)": "14416 or 1-800-891-4416",
    "Snehi": "+91 44-24640050",
}

CRISIS_MESSAGE = (
    "I'm really sorry you're feeling this overwhelmed right now.\n\n"
    "🇮🇳 **India Mental Health Helplines:**\n"
    + "\n".join([f"📞 {name}: {number}" for name, number in CRISIS_HELPLINES.items()])
    + "\n\nPlease reach out to a trusted person or mental health professional immediately.\n\n"
    "You are not alone. 💙"
)

# ──────────────────────────────────────────────
# CRISIS KEYWORDS
# ──────────────────────────────────────────────
CRISIS_KEYWORDS = [
    "suicide",
    "kill myself",
    "end my life",
    "don't want to live",
    "want to disappear",
    "self harm",
    "hurt myself",
    "i want to die",
    "no reason to live",
    "can't go on"
]

# ──────────────────────────────────────────────
# CBT TIPS (per emotion)
# ──────────────────────────────────────────────
CBT_TIPS = {
    "sadness": [
        "🧠 **CBT Tip:** Write down one negative thought you're having. Then ask — is this thought 100% true? What evidence supports or challenges it?",
        "🧠 **CBT Tip:** Behavioural activation — even a 10-minute walk can interrupt a low mood cycle.",
        "🧠 **CBT Tip:** Try naming 3 small things you did today, no matter how minor. Sadness shrinks our sense of achievement.",
    ],
    "fear": [
        "🧠 **CBT Tip:** Write down your worst-case scenario. Then write the most realistic outcome. Usually reality sits closer to realistic than worst-case.",
        "🧠 **CBT Tip:** Break the thing you're anxious about into the smallest possible next step. Just the next step, nothing more.",
        "🧠 **CBT Tip:** Anxiety thrives on avoidance. Facing small fears gradually reduces their power over time.",
    ],
    "anger": [
        "🧠 **CBT Tip:** Before reacting, ask — will this matter in 5 days? If not, it may not deserve 5 minutes of your energy now.",
        "🧠 **CBT Tip:** Anger often masks another emotion underneath — hurt, fear, or disappointment. Try to identify the root feeling.",
        "🧠 **CBT Tip:** Physical movement (a walk, stretching) helps discharge anger energy from the body faster than thinking it through.",
    ],
    "joy": [
        "🧠 **CBT Tip:** Savour this. Research shows deliberately pausing to notice positive moments strengthens long-term emotional resilience.",
        "🧠 **CBT Tip:** Share this feeling with someone — expressing positive emotions amplifies them.",
    ],
    "neutral": [
        "🧠 **CBT Tip:** Check in with yourself — sometimes 'neutral' is suppressed emotion. What's one word that describes how you're really doing?",
        "🧠 **CBT Tip:** Journalling for even 5 minutes a day builds emotional self-awareness over time.",
    ],
    "disgust": [
        "🧠 **CBT Tip:** Disgust often signals a values violation. Ask yourself — what value of mine feels threatened here?",
    ],
    "surprise": [
        "🧠 **CBT Tip:** Unexpected events can destabilise us. Give yourself permission to feel unsettled before jumping to problem-solving.",
    ],
}

# ──────────────────────────────────────────────
# GROUNDING EXERCISES
# ──────────────────────────────────────────────
GROUNDING_EXERCISES = {
    "5-4-3-2-1": (
        "**5-4-3-2-1 Grounding Exercise:**\n"
        "- 👁️ Name **5 things** you can see right now\n"
        "- ✋ Name **4 things** you can physically feel\n"
        "- 👂 Name **3 things** you can hear\n"
        "- 👃 Name **2 things** you can smell\n"
        "- 👅 Name **1 thing** you can taste\n\n"
        "This brings your mind back to the present moment."
    ),
    "box_breathing": (
        "**Box Breathing (4-4-4-4):**\n"
        "- Inhale for **4 seconds**\n"
        "- Hold for **4 seconds**\n"
        "- Exhale for **4 seconds**\n"
        "- Hold for **4 seconds**\n\n"
        "Repeat 4 times. Activates the parasympathetic nervous system and reduces anxiety."
    ),
    "body_scan": (
        "**Quick Body Scan:**\n"
        "Close your eyes. Starting from the top of your head, slowly move attention down to your feet.\n"
        "Notice any tension without trying to fix it. Just observe.\n\n"
        "Takes 2 minutes. Reduces physical tension caused by stress."
    ),
    "cold_water": (
        "**Grounding with Cold Water:**\n"
        "Splash cold water on your face or hold your wrists under cold running water for 30 seconds.\n\n"
        "Activates the dive reflex — slows heart rate and reduces emotional intensity quickly."
    ),
}

# ──────────────────────────────────────────────
# EMOTION → GROUNDING EXERCISE MAPPING
# ──────────────────────────────────────────────
EMOTION_GROUNDING_MAP = {
    "fear": "box_breathing",
    "anger": "cold_water",
    "sadness": "body_scan",
    "neutral": "5-4-3-2-1",
    "disgust": "body_scan",
    "surprise": "box_breathing",
    "joy": None,  # no grounding needed for joy
}

def get_grounding_for_emotion(emotion: str) -> str | None:
    """Returns the grounding exercise text for a given emotion, or None for joy."""
    key = EMOTION_GROUNDING_MAP.get(emotion)
    if key:
        return GROUNDING_EXERCISES[key]
    return None
