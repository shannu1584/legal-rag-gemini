import os
from flask import Flask, request, render_template, send_file
from dotenv import load_dotenv
from google import genai
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from datetime import datetime
from sentence_transformers import SentenceTransformer

# =====================================
# LOAD ENV
# =====================================
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={"api_version": "v1"}  # VERY IMPORTANT
)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
app = Flask(__name__)

# =====================================
# GLOBALS
# =====================================
index = None
chunks = []
history = []
uploaded_flag = False
current_filename = "Uploaded Document"


# =====================================
# EMBEDDING FUNCTION
# =====================================
def get_embedding(text):
    return embedding_model.encode(text, convert_to_numpy=True).astype("float32")

# =====================================
# BUILD FAISS INDEX
# =====================================
def build_retriever(pdf_path):
    global index, chunks

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents(documents)
    chunks = [doc.page_content for doc in split_docs]

    embeddings = [get_embedding(chunk) for chunk in chunks]

    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    print("✅ Document indexed with local embeddings (MiniLM)")


# =====================================
# RAG ANSWER
# =====================================
def rag_answer(question):
    global index, chunks

    if index is None:
        return "Please upload a PDF first."

    question_embedding = get_embedding(question)
    D, I = index.search(np.array([question_embedding]), k=5)

    retrieved_chunks = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved_chunks)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
Answer briefly in 1-2 sentences.

Context:
{context}

Question:
{question}
"""
    )

    return response.text.strip()


def generate_summary():
    global chunks

    if not chunks:
        return "Upload PDF first."

    combined_text = "\n".join(chunks[:12])

    prompt = f"""
Return ONLY in the exact format below.
Do NOT write any introduction.
Do NOT add extra explanation.
Start directly with SUMMARY:

SUMMARY:
• 8 bullet points

OBLIGATIONS:
• 8 bullet points

RISKS:
• 5 bullet points

Use simple English.

Text:
{combined_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()
# ===============================
# CONFIDENCE CALCULATION
# ===============================
def calculate_confidence():
    global chunks

    if not chunks:
        return "0%"

    # Basic logic based on document depth
    score = min(70 + len(chunks) // 2, 97)

    return f"{score}%"
# =====================================
# ROUTES
# =====================================
@app.route("/", methods=["GET", "POST"])
def home():
    global uploaded_flag, current_filename

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file.save(file.filename)
            build_retriever(file.filename)
            uploaded_flag = True
            current_filename = file.filename

    return render_template("index.html", uploaded=uploaded_flag)


@app.route("/chat", methods=["GET", "POST"])
def chat():
    global history

    if request.method == "POST":
        q = request.form["question"]
        a = rag_answer(q)
        history.append((q, a))

    return render_template("chat.html", history=history)


@app.route("/summary")
def summary():
    output = generate_summary()

    parts = output.split("OBLIGATIONS:")
    summary_part = parts[0].replace("SUMMARY:", "").strip()

    obligations_part = ""
    risks_part = ""

    if len(parts) > 1:
        sub = parts[1].split("RISKS:")
        obligations_part = sub[0].strip()
        if len(sub) > 1:
            risks_part = sub[1].strip()

    return render_template(
        "summary.html",
        summary=summary_part,
        obligations=obligations_part,
        risks=risks_part,
        filename=current_filename,
        confidence = calculate_confidence()
    )


@app.route("/export_pdf")
def export_pdf():
    text = generate_summary()

    filename = "Legal_Report.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.darkblue,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.red,
        spaceBefore=15,
        spaceAfter=10
    )

    normal_style = ParagraphStyle(
        'NormalStyle',
        parent=styles['BodyText'],
        fontSize=12,
        leading=16
    )

    elements = []

    # Title
    elements.append(Paragraph("Legal Document Analysis Report", title_style))
    elements.append(Spacer(1, 12))

    # Document Name
    elements.append(Paragraph(f"<b>Document:</b> {current_filename}", normal_style))
    elements.append(Spacer(1, 6))

    # Generated Date
    today = datetime.now().strftime("%d %B %Y")
    elements.append(Paragraph(f"<b>Generated On:</b> {today}", normal_style))
    elements.append(Spacer(1, 20))

    # Split sections
    sections = text.split("OBLIGATIONS:")
    summary_part = sections[0].replace("SUMMARY:", "").strip()

    obligations_part = ""
    risks_part = ""

    if len(sections) > 1:
        sub = sections[1].split("RISKS:")
        obligations_part = sub[0].strip()
        if len(sub) > 1:
            risks_part = sub[1].strip()

    # SUMMARY
    elements.append(Paragraph("SUMMARY", heading_style))
    elements.append(Spacer(1, 10))

    for line in summary_part.split("\n"):
        if line.strip():
            elements.append(Paragraph(f"• {line.strip()}", normal_style))
            elements.append(Spacer(1, 5))

    # OBLIGATIONS
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("OBLIGATIONS", heading_style))
    elements.append(Spacer(1, 10))

    for line in obligations_part.split("\n"):
        if line.strip():
            elements.append(Paragraph(f"• {line.strip()}", normal_style))
            elements.append(Spacer(1, 5))

    # RISKS
    elements.append(Spacer(1, 15))
    elements.append(Paragraph("RISKS", heading_style))
    elements.append(Spacer(1, 10))

    for line in risks_part.split("\n"):
        if line.strip():
            elements.append(Paragraph(f"• {line.strip()}", normal_style))
            elements.append(Spacer(1, 5))

    doc.build(elements)

    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
