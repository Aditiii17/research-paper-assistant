# 1. Install required libraries
#!pip install -q transformers sentence-transformers faiss-cpu PyMuPDF gradio

# 2. Import necessary packages
import gradio as gr
import fitz  # PyMuPDF
import faiss
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# 3. PDF text extraction
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "".join([page.get_text() for page in doc])

# 4. Create FAISS index
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_faiss_index(texts):
    embeddings = embedder.encode(texts)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, texts

# 5. Summarizer & QA Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0 if torch.cuda.is_available() else -1)


doc_chunks, faiss_index, original_text = [], None, ""

# 6. Text splitter
def split_text(text, max_tokens=400):
    paras = text.split("\n")
    chunks, chunk = [], ""
    for para in paras:
        words = (chunk + " " + para).split()
        if len(words) <= max_tokens:
            chunk = " ".join(words)
        else:
            chunks.append(chunk.strip())
            chunk = para
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# 7. Upload handler
def upload_handler(file):
    global doc_chunks, faiss_index, original_text
    original_text = extract_text_from_pdf(file.name)
    doc_chunks = split_text(original_text)
    faiss_index, _, _ = create_faiss_index(doc_chunks)
    return "✅ Document uploaded and indexed successfully."

# 8. Summarization logic (with progress)
def summarize_doc(progress=gr.Progress(track_tqdm=True)):
    global original_text
    if not original_text:
        return "❌ No document uploaded."

    chunks = split_text(original_text, max_tokens=400)

    summaries = []
    for i, chunk in enumerate(progress.tqdm(chunks, desc="Summarizing")):
        try:
            summary = summarizer(
                chunk,
                max_length=200,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"[Error summarizing chunk {i+1}: {str(e)}]")

    return "\n\n".join(summaries)

# 9. Q&A logic
def answer_question(question):
    global doc_chunks, faiss_index
    if faiss_index is None:
        return "❌ Please upload a document first."
    q_embed = embedder.encode([question])
    D, I = faiss_index.search(np.array(q_embed), k=3)
    context = " ".join([doc_chunks[i] for i in I[0]])
    return qa_pipeline({'question': question, 'context': context})['answer']

# 10. Gradio UI (redesigned layout)
with gr.Blocks(css="""
.gr-box {
    border-radius: 12px;
    padding: 20px;
    background-color: #ffffff;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}
textarea, input[type='text'] {
    font-size: 16px;
}
button {
    font-size: 16px;
    padding: 10px 16px;
}
""") as demo:
    gr.Markdown("""
    # 📚 AI Research Assistant
    Upload your PDF research paper to get a clean humanized summary and ask questions directly from its content.
    """)

    with gr.Column(elem_classes="gr-box"):
        pdf_input = gr.File(label="📄 Upload PDF", file_types=[".pdf"])
        upload_btn = gr.Button("📥 Index Document")
        upload_status = gr.Textbox(label="Status", interactive=False)

    with gr.Column(elem_classes="gr-box"):
        summarize_btn = gr.Button("🧠 Summarize Document")
        summary_output = gr.Textbox(label="Document Summary", lines=15, interactive=False)

    with gr.Column(elem_classes="gr-box"):
        question_input = gr.Textbox(label="❓ Ask a Question")
        answer_btn = gr.Button("🔍 Get Answer")
        answer_output = gr.Textbox(label="Answer", lines=4, interactive=False)

    upload_btn.click(upload_handler, inputs=pdf_input, outputs=upload_status)
    summarize_btn.click(summarize_doc, outputs=summary_output)
    answer_btn.click(answer_question, inputs=question_input, outputs=answer_output)

    demo.queue().launch()
