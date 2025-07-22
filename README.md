

# 📘 Research Paper Assistant

An intelligent, AI-powered assistant for reading, summarizing, and querying research papers (PDFs). Upload any academic document, get a clean and humanized summary, and ask questions about the content — all in one lightweight and free tool.

---

## 🚀 Features

- 📄 Upload and parse full academic PDF papers
- 🧠 Generate **humanized summaries** (no rigid sectioning)
- ❓ Ask **contextual questions** about the paper
- ⚡ Real-time document embedding + FAISS-powered search
- 🖥️ Runs on **Hugging Face Spaces**, **Google Colab**, or **locally**
- 💸 Fully **free** and uses only **open-source models**

---

## 🧰 Tech Stack

| Technology / Library                    | Purpose                                                               |
|----------------------------------------|-----------------------------------------------------------------------|
| `Gradio`                                | Web interface for upload, summarization, and Q&A                      |
| `PyMuPDF (fitz)`                        | Extract plain text from PDF files                                     |
| `Transformers` (Hugging Face)           | Load and run pre-trained models for summarization and QA              |
| `SentenceTransformers`                  | Convert text and questions to semantic vector representations         |
| `FAISS`                                 | Perform fast and efficient semantic search using embeddings           |
| `Torch`                                 | Backend engine for inference (runs on CPU or GPU if available)        |

---

## 🗂 Project Structure

```

research-paper-assistant/
├── app.py                # Main app logic with Gradio interface
├── requirements.txt      # List of dependencies to run the app

````

---

## 🧠 Models Used (Free & Open Source)

| Task            | Model ID                                      |
|-----------------|-----------------------------------------------|
| Summarization   | `facebook/bart-large-cnn`                     |
| Question Answer | `deepset/roberta-base-squad2`                 |
| Embedding       | `sentence-transformers/all-MiniLM-L6-v2`      |

All models are hosted publicly on [Hugging Face Hub](https://huggingface.co/models) and downloaded automatically.

---

## 🔎 Functionality Breakdown

### 📤 PDF Upload & Text Extraction
- Extracts full text using `PyMuPDF`
- Stores the plain text for downstream tasks

### 🧩 Text Chunking
- Chunks long documents intelligently to handle model input size limits
- Keeps context and sentence boundaries intact

### 🧠 Summarization
- Uses `facebook/bart-large-cnn` to summarize long-form academic text
- Automatically merges summaries into one human-friendly response

### 🧬 Semantic Embedding & Vector Search
- Embeds chunks using `MiniLM`
- Uses FAISS to retrieve most relevant text blocks for a query

### 🤖 Q&A Engine
- Retrieves top context chunks
- Applies `deepset/roberta-base-squad2` to answer natural-language questions

---

## 🖥️ User Interface

Built with **Gradio**, the app provides:

- 📤 Upload button for PDF input
- 🧠 "Summarize" button with progress tracking
- ❓ Question input for real-time Q&A
- 📄 Display boxes for summary and answers

---

## 📦 Installation

### 🔧 Requirements

Python ≥ 3.9, pip

### 💻 Local Setup

```bash
git clone https://github.com/your-username/research-paper-assistant
cd research-paper-assistant
pip install -r requirements.txt
python app.py
````

### ✅ Run on Colab

Just paste the script into [Google Colab](https://colab.research.google.com/) and run — no setup needed.

### 🌐 Deploy on Hugging Face

* Create a new [Hugging Face Space](https://huggingface.co/spaces)
* Choose `Gradio` as the SDK
* Upload:

  * `app.py`
  * `requirements.txt`
* Click **Deploy**

---

## 🔄 Roadmap

* [ ] Add download option for summary
* [ ] Persistent Q\&A chat window
* [ ] Offline LLM support with GGUF
* [ ] Export results to Notion or Markdown

---

## 👨‍🏫 Ideal For

* Students working on thesis/lit reviews
* Researchers validating prior work
* Anyone needing fast insights from academic documents

---

## 📃 License

MIT License — free for personal and commercial use.

---

> Built with 🧠 by combining Hugging Face, FAISS, Gradio, and pure passion for AI in education.


