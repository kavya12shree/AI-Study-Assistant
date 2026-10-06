
import streamlit as st
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
import ollama

# -----------------------------
# Page settings
# -----------------------------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="centered"
)

# -----------------------------
# Professional CSS
# -----------------------------
st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f5f7ff, #eef2ff);
    }

    /* Main content */
    .main .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    h1 {
        text-align: center;
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #1e293b;
        margin-bottom: 10px;
    }

    /* Subtitle */
    .stApp p {
        color: #475569;
    }

    /* Section headings */
    h2, h3 {
        color: #1e293b !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #6366f1;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }

    /* Text input */
    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        padding: 12px;
        font-size: 16px;
    }

    .stTextInput input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: 600;
        background: #4f46e5;
        color: white;
        transition: 0.2s;
    }

    .stButton > button:hover {
        background: #3730a3;
        color: white;
        transform: translateY(-1px);
    }

    /* Success messages */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* Answer box */
    .answer-box {
        background: white;
        border-left: 5px solid #4f46e5;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.06);
    }

    /* Source box */
    .source-box {
        background: #f8fafc;
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 12px;
        border: 1px solid #e2e8f0;
    }

    /* Divider */
    hr {
        margin-top: 25px;
        margin-bottom: 25px;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.title("📚 AI Study Assistant")

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Upload your study material and ask questions using AI."
    "</p>",
    unsafe_allow_html=True
)

# -----------------------------
# ChromaDB
# -----------------------------
client = chromadb.Client()

collection = client.get_or_create_collection(
    name="study_material"
)

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "📄 Upload your study material",
    type=["pdf"]
)

if uploaded_file:

    # -----------------------------
    # Extract PDF text
    # -----------------------------
    reader = PdfReader(uploaded_file)

    full_text = ""

    for page in reader.pages:
        text = page.extract_text()

        if text:
            full_text += text + "\n"

    st.success("✅ PDF text extracted successfully!")

    st.write("📄 Pages:", len(reader.pages))

    # -----------------------------
    # Split text into chunks
    # -----------------------------
    chunk_size = 500

    chunks = []

    for i in range(0, len(full_text), chunk_size):

        chunk = full_text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    st.write("📝 Number of chunks:", len(chunks))

    # -----------------------------
    # Load embedding model
    # -----------------------------
    with st.spinner("Loading embedding model..."):

        model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    st.success("✅ Embedding model loaded!")

    # -----------------------------
    # Create embeddings
    # -----------------------------
    with st.spinner("Creating embeddings..."):

        embeddings = model.encode(
            chunks
        ).tolist()

    st.success("✅ Embeddings created!")

    # -----------------------------
    # Store in ChromaDB
    # -----------------------------
    collection.upsert(
        documents=chunks,
        embeddings=embeddings,
        ids=[
            f"{uploaded_file.name}_{i}"
            for i in range(len(chunks))
        ]
    )

    st.success("✅ PDF content stored in ChromaDB!")

    # ==================================================
    # QUESTION SECTION
    # ==================================================

    st.divider()

    st.header("💬 Ask a Question")

    question = st.text_input(
        "Type your question about the PDF:"
    )

    # -----------------------------
    # Ask button
    # -----------------------------
    if st.button("🔍 Ask"):

        if not question.strip():

            st.warning("⚠️ Please enter a question.")

        else:

            # -----------------------------
            # Create question embedding
            # -----------------------------
            question_embedding = model.encode(
                question
            ).tolist()

            # -----------------------------
            # Search ChromaDB
            # -----------------------------
            results = collection.query(
                query_embeddings=[question_embedding],
                n_results=3
            )

            relevant_chunks = results["documents"][0]

            # Combine retrieved chunks
            context = "\n\n".join(
                relevant_chunks
            )

            # -----------------------------
            # Create prompt for Gemma
            # -----------------------------
            prompt = f"""
You are an AI Study Assistant.

Answer the question using ONLY the information
provided in the uploaded PDF.

If the answer is not present in the PDF,
say:

"I could not find the answer in the uploaded material."

Uploaded PDF content:
{context}

Question:
{question}

Answer:
"""

            # -----------------------------
            # Generate answer using Ollama
            # -----------------------------
            with st.spinner("🤖 Generating answer..."):

                response = ollama.chat(
                    model="gemma3:1b",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

            answer = response["message"]["content"]

            # -----------------------------
            # Display answer
            # -----------------------------
            st.subheader("🤖 Answer")

            st.markdown(
                f"""
                <div class="answer-box">
                    {answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("📚 Source")

            st.markdown(
                f"""
                <div class="source-box">
                    📄 <b>{uploaded_file.name}</b>
                </div>
                """,
                unsafe_allow_html=True
            )

    # -----------------------------
    # Show extracted text
    # -----------------------------
    with st.expander("📖 View Extracted Text"):

        st.write(full_text)

