
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

---

## 💬 Sample Queries to Try

- What pest-control methods are recommended for paddy in Tamil Nadu?
- How to manage drought stress in groundnut cultivation?
- What issues do sugarcane farmers in Maharashtra commonly face?
- What are best sowing times for rabi wheat in Punjab?
- How to control leaf spot disease in chillies?
- When should cotton be irrigated in Karnataka?

---

## 📹 Demo Video Instructions

Record a 3–5 minute screencast demonstrating:

1. Local launch of the app and Ollama model.
2. Preprocessing and embedding of KCC data.
3. At least 3 queries that return offline/local answers.
4. At least 2 queries that fall back to SerpAPI and show Google-based results.
5. Upload to Google Drive and provide a view-only link.

---

## ✅ Submission Checklist

- [x] Raw and preprocessed KCC datasets
- [x] Vector DB (FAISS) setup
- [x] LLM integration via Ollama
- [x] Query handling with fallback
- [x] Sample queries (min. 10)
- [x] Screencast video
- [x] README with instructions

---

## 📚 Acknowledgements

- [Kisan Call Center Dataset](https://data.gov.in)
- [Ollama](https://ollama.com)
- [Gemma LLM by Google](https://ai.google.dev)
- [LangChain](https://www.langchain.com/)
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS by Meta AI](https://github.com/facebookresearch/faiss)
- [SerpAPI](https://serpapi.com/)
