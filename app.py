import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ==========================================
# STREAMLIT PAGE
# ==========================================

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about it using Ollama.")


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


with st.spinner("Loading embedding model..."):

    embedding_model = load_embedding_model()


# ==========================================
# CREATE CHROMADB
# ==========================================

@st.cache_resource
def create_chroma():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    return client, collection


client, collection = create_chroma()


# ==========================================
# CREATE CHUNKS
# ==========================================

def create_chunks(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunk = text[i:i + chunk_size]

        if chunk.strip():

            chunks.append(chunk)

    return chunks


# ==========================================
# PDF UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type=["pdf"]
)


# ==========================================
# PROCESS PDF
# ==========================================

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Reading PDF..."):

            reader = PdfReader(uploaded_file)

            text = ""

            for page_number, page in enumerate(reader.pages):

                page_text = page.extract_text()

                if page_text:

                    text += page_text + "\n"


        # ==================================
        # CHECK TEXT
        # ==================================

        if not text.strip():

            st.error(
                "❌ No text could be extracted from this PDF. "
                "The PDF may be scanned or image-based."
            )

            st.stop()


        st.success(
            f"✅ PDF read successfully! "
            f"Pages: {len(reader.pages)}"
        )

        st.write(
            f"**Characters extracted:** {len(text)}"
        )


        # ==================================
        # CREATE CHUNKS
        # ==================================

        chunks = create_chunks(text)

        st.write(
            f"**Number of chunks:** {len(chunks)}"
        )


        # ==================================
        # GENERATE EMBEDDINGS
        # ==================================

        with st.spinner(
            "Generating document embeddings..."
        ):

            embeddings = embedding_model.encode(
                chunks
            )


        st.success("✅ Embeddings generated!")


        # ==================================
        # STORE IN CHROMADB
        # ==================================

        with st.spinner(
            "Storing document in ChromaDB..."
        ):

            # Create unique IDs
            ids = []

            for i in range(len(chunks)):

                ids.append(
                    f"{uploaded_file.name}_chunk_{i}"
                )


            # Use upsert instead of add
            # to avoid duplicate ID errors

            collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings.tolist()
            )


        st.success(
            "✅ PDF stored successfully in ChromaDB!"
        )

        # Save information in session
        st.session_state["pdf_name"] = uploaded_file.name

        st.session_state["pdf_processed"] = True


# ==========================================
# QUESTION SECTION
# ==========================================

if st.session_state.get(
    "pdf_processed",
    False
):

    st.divider()

    st.subheader("💬 Ask a Question")

    question = st.text_input(
        "Enter your question:"
    )


    # ======================================
    # ASK BUTTON
    # ======================================

    if st.button("🔍 Ask AI"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            # ==================================
            # CREATE QUESTION EMBEDDING
            # ==================================

            with st.spinner(
                "Searching the document..."
            ):

                question_embedding = (
                    embedding_model.encode(
                        [question]
                    )
                )


                # ==================================
                # SEARCH CHROMADB
                # ==================================

                results = collection.query(
                    query_embeddings=
                    question_embedding.tolist(),

                    n_results=5
                )


                # ==================================
                # GET RELEVANT CHUNKS
                # ==================================

                relevant_chunks = (
                    results["documents"][0]
                )


                # ==================================
                # COMBINE CONTEXT
                # ==================================

                context = "\n\n".join(
                    relevant_chunks
                )


            # ==================================
            # CREATE PROMPT
            # ==================================

            prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY
the information provided in the document context.

If the answer is not present in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""


            # ==================================
            # ASK OLLAMA
            # ==================================

            with st.spinner(
                "Generating answer..."
            ):

                try:

                    response = ollama.chat(

                        model="llama3.2",

                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ]
                    )


                    answer = (
                        response["message"]["content"]
                    )


                    # ==================================
                    # DISPLAY ANSWER
                    # ==================================

                    st.subheader("🤖 AI Answer")

                    st.write(answer)


                    # ==================================
                    # SHOW SOURCES
                    # ==================================

                    with st.expander(
                        "📖 View relevant document sections"
                    ):

                        for i, chunk in enumerate(
                            relevant_chunks
                        ):

                            st.markdown(
                                f"**Section {i + 1}**"
                            )

                            st.write(chunk)

                            st.divider()


                except Exception as e:

                    st.error(
                        f"❌ Ollama error: {e}"
                    )

                    st.info(
                        "Make sure Ollama is running "
                        "and the llama3.2 model is installed."
                    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("⚙️ Setup")

    st.write(
        "This application uses:"
    )

    st.write("📄 PDF Reader")
    st.write("🧠 Sentence Transformers")
    st.write("🗄️ ChromaDB")
    st.write("🤖 Ollama + Llama 3.2")

    st.divider()

    st.write(
        "Upload a PDF, process it, "
        "and then ask questions."
    )
