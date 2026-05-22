# 🧠 EmpathAI — Transformer-Based Empathetic Mental Health Chatbot

> An NLP-powered mental health support chatbot that detects your emotional state using a fine-tuned Transformer model and responds with **your own words** woven into CBT-informed replies — moving beyond generic template chatbots.

🌐 **[Live Demo](https://empathy-chatbot-t5q6wgik4v9zypfybutpci.streamlit.app)** &nbsp;|&nbsp; 📄 **Prototype v1** &nbsp;|&nbsp; 🐍 Python 3.11+

---

## 👥 Team

**Rounak Chaudhury · Aharshi Sinha · Arkapriya Nandi**

---

## 🌟 What Makes This Different

Most chatbots return the same scripted line regardless of what you said. EmpathAI uses a genuine NLP pipeline:

- **Transformer-based emotion detection** — fine-tuned DistilRoBERTa classifies your message into 7 emotion categories
- **Confidence thresholding** — if the model isn't sure (< 45%), a keyword fallback layer takes over
- **Snippet extraction** — your exact words are pulled from your message and injected into the response
- **CBT-informed replies** — every response uses real Cognitive Behavioural Therapy techniques
- **Crisis safety filter** — checks every message before any NLP runs; shows helplines immediately if needed

---

## ⚙️ How It Works — The 4-Stage Pipeline

```
User Input
    ↓
[Stage 1] Crisis Safety Filter         ← 16 crisis phrases checked first
    ↓ (if safe)
[Stage 2] Emotion Detection            ← DistilRoBERTa (confidence ≥ 45%)
               ↓ (if low confidence)        or Keyword Fallback (< 45%)
[Stage 3] Snippet Extraction           ← your words pulled from the message
    ↓
[Stage 4] CBT Response Generation      ← snippet injected into template
    ↓
Display
```

### Emotion Classes
`sadness` · `fear` · `anger` · `joy` · `disgust` · `surprise` · `neutral`

### CBT Techniques Used
- Socratic questioning
- Grounding exercises (for anxiety/fear)
- Cognitive restructuring prompts
- Behavioural activation cues (for sadness)
- Validation before advice — always

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Frontend / UI | Streamlit 1.35+ |
| Emotion Classifier | `j-hartmann/emotion-english-distilroberta-base` (HuggingFace) |
| NLP Framework | HuggingFace Transformers 4.40+ |
| Deep Learning | PyTorch 2.2+ (CPU inference) |
| Deployment | Streamlit Community Cloud |
| Language | Python 3.11+ |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### Local Development

```bash
# 1. Clone the repo
git clone https://github.com/Rounak5106/empathy-chatbot.git

# 2. Move into the directory
cd empathy-chatbot

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> On first run, the DistilRoBERTa model weights (~330 MB) download automatically from HuggingFace Hub and cache locally. Subsequent runs load instantly.

### Deployment Files

| File | Purpose |
|---|---|
| `app.py` | Main application — all NLP logic, UI, session state |
| `requirements.txt` | Python deps: streamlit, transformers, torch, sentencepiece |
| `packages.txt` | System dep: `libgomp1` (required by PyTorch on Linux) |

---

## ⚠️ Limitations

This is an **academic prototype**, not a clinical tool.

- Model trained on social-media text — therapeutic language behaves differently (domain shift)
- No memory across conversation turns — each message classified independently
- Negation handling is unreliable without fine-tuning
- Crisis filter uses string matching only — metaphorical or indirect expressions may not trigger it
- No pathway to escalate to a human professional

---

## 🔭 Roadmap

| Priority | Feature |
|---|---|
| 🔴 High | LLM integration (replace templates with Claude Haiku / GPT-4o-mini) |
| 🔴 High | Domain fine-tuning on EmpatheticDialogues dataset |
| 🔴 High | Empirical confidence threshold calibration |
| 🟡 Medium | RAG pipeline over a CBT knowledge base (FAISS + sentence-transformers) |
| 🟡 Medium | Multi-turn context window for session-aware responses |
| 🟡 Medium | Hindi and other Indian language support |
| 🟢 Future | Voice input support |
| 🟢 Future | Personalized conversation memory |

---

## 🛡️ Crisis Resources (India)

If you or someone you know is in crisis:

| Helpline | Number |
|---|---|
| iCall | 9152987821 |
| Vandrevala Foundation | 1860-2662-345 |
| AASRA | 9820466627 |

---

## 📚 References

- Hartmann, J. (2022). [emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base). HuggingFace Model Hub.
- Rashkin et al. (2019). Towards Empathetic Open-Domain Conversation Models. ACL 2019.
- Sanh et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.

---

## 📄 License

This project is an academic prototype. Not intended for clinical or production use.

> ⚕️ **Disclaimer:** EmpathAI is not a substitute for professional mental health care. If you are experiencing a mental health crisis, please contact a qualified professional or the helplines listed above.
