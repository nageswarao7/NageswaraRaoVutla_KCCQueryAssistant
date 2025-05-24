
# 🌾 KCC Query Assistant (Offline + Fallback)

This is a local-first AI application that allows users to query agricultural advice from the **Kisan Call Center (KCC)** dataset using a local LLM (Gemma 2B via Ollama) and semantic search (FAISS). If no local context is found, it gracefully falls back to live internet search using SerpAPI.

---

## 🔧 Features

- ✅ Offline semantic search with FAISS and HuggingFace Embeddings
- ✅ Local LLM via Ollama (Gemma 2B)
- ✅ Preprocessing of real agricultural Q&A data (KCC)
- ✅ Metadata-preserved document chunking
- ✅ Streamlit UI with structured display
- ✅ Fallback to SerpAPI search if no relevant local context is found

---

## 🛠️ Installation

```bash
git clone https://github.com/nageswarao7/NageswaraRaoVutla_KCCQueryAssistant.git
cd KCCQueryAssistant
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. **Start Ollama and ensure Gemma is installed:**

   ```bash
   ollama run gemma:2b
   ```

2. **Run Streamlit app:**

   ```bash
   streamlit run app.py
   ```

---

## 🔐 Streamlit Secrets

Create a file named `.streamlit/secrets.toml`:

```toml
SERPAPI_API_KEY = "your_serpapi_api_key"
```

---

## 📁 Project Structure

```
├── app.py                    # Streamlit UI
├── utils.py                  # Preprocessing, FAISS, and LLM functions
├── sampleeeee.csv            # Raw dataset
├── processed_docs.json       # Preprocessed Q&A chunks
├── faiss_index.pkl           # FAISS index
├── requirements.txt
├── .streamlit/
│   └── secrets.toml          # Your SerpAPI key
└── README.md
```
