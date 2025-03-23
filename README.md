# 🚀 Building a Chat Interface with Groq, Gradio, and Langfuse

This project helps you build a chat interface that connects with Groq's LLMs, provides a user-friendly UI via Gradio, and enables rich observability using Langfuse. Whether you're experimenting with LLMs or deploying them in production, this setup is designed to give you insight into model performance and user behavior.

---

## 🔍 Why LLM Observability?

Understanding how your models perform in the wild is critical. Observability helps you answer questions like:
- Are responses actually useful?
- Which model is best for which task?
- What patterns emerge in user conversations?
- Where can you improve the experience?

---

## 🛠️ Why Langfuse?

[Langfuse](https://langfuse.com) is an open-source observability platform tailored for LLMs. It provides:

- ✅ Full tracing of user interactions  
- ✅ Feedback collection  
- ✅ Monitoring of latency, token usage, and costs  
- ✅ Session-based grouping  
- ✅ Self-hosting support for data control  

---

## 💬 What You’ll Build

- AI Chat interface  
- Model selection (supports multiple Groq models)  
- Feedback buttons (👍 / 👎)  
- End-to-end traceability in Langfuse  

---

## ⚙️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/groq-langfuse-chat.git
cd groq-langfuse-chat
```

### 2. Set Up the Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your API Keys
The repo includes a .env template. Update it with your credentials:
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...    # from cloud.langfuse.com
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com
GROQ_API_KEY=gsk_...             # from console.groq.com
```

### ▶️ Run the App

```bash
python ai_app.py
```
Access it at: http://localhost:7860

## 🧪 How to Use
- Choose a model from the dropdown
- Start chatting - each response is traced in Langfuse

## 📊 Monitor in Langfuse
- Visit: cloud.langfuse.com
- Head to the Traces tab
- Inspect model usage, response time, user feedback, and session history

## 🚀 What to Explore
- Model Comparison: Try different models for the same prompt
- Session Tracking: Analyze conversation history and flow
- User Feedback: Use scores to improve responses

## Conclusion

With Groq’s powerful LLMs, Gradio’s UI, and Langfuse’s observability - you’ve got a solid foundation for building, testing, and improving AI applications.

### Next Steps
- Add custom prompt templates
- Implement advanced error handling
- Introduce more model controls and temperature settings
