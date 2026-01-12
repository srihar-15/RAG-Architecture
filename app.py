import streamlit as st
import os
import time
import cohere
from google import genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pypdf import PdfReader

st.set_page_config(page_title="RAG Architect | Advanced", layout="wide")
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if "vectors_uploaded" not in st.session_state:
    st.session_state.vectors_uploaded = False

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)

@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-mpnet-base-v2')

@st.cache_resource
def get_cohere_client():
    return cohere.Client(COHERE_API_KEY)

def configure_gemini():
    return genai.Client()


def process_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.create_documents([text])
    return chunks

def format_docs_for_prompt(docs):
    formatted = ""
    for i, doc in enumerate(docs):
        content = doc if isinstance(doc, str) else doc.payload.get('text', '')
        content = content.replace('\n', ' ')
        formatted += f"Source [{i+1}]: {content}\n\n"
    return formatted


st.title("🧬 Advanced RAG Architect")
st.markdown("### Powered by Pinecone, Cohere, and Gemini")

with st.sidebar:
    st.header("1. Ingestion Pipeline")
    
    uploaded_file = st.file_uploader("Option A: Upload PDF", type="pdf")
    
    raw_text_input = st.text_area("Option B: Paste Text", height=150, placeholder="Paste article or notes here...")
    
    if st.button("Ingest Data"):
        if not uploaded_file and not raw_text_input:
            st.warning("Please upload a file or paste text first.")
        else:
            with st.spinner("Processing Pipeline..."):
                all_chunks = []
                
                if uploaded_file:
                    text_pdf = process_pdf(uploaded_file)
                    chunks_pdf = chunk_text(text_pdf)
                    for c in chunks_pdf:
                        c.metadata = {"source": uploaded_file.name}
                    all_chunks.extend(chunks_pdf)
                    st.info(f"Processed PDF: {len(chunks_pdf)} chunks")

                if raw_text_input:
                    chunks_text = chunk_text(raw_text_input)
                    for c in chunks_text:
                        c.metadata = {"source": "User_Type_Input"}
                    all_chunks.extend(chunks_text)
                    st.info(f"Processed Raw Text: {len(chunks_text)} chunks")
                
                if all_chunks:
                    index = get_pinecone_index()
                    try:
                        index.delete(delete_all=True)
                        st.toast("🧹 Cleared old index data.", icon="🗑️")
                        time.sleep(1) 
                    except Exception as e:
                        st.warning(f"Could not clear index: {e}")

                    model = get_embedding_model()
                    embeddings = model.encode([c.page_content for c in all_chunks])
                    
                    vectors = []
                    for i, chunk in enumerate(all_chunks):
                        safe_id = f"{chunk.metadata['source']}_{i}_{int(time.time())}" 
                        vectors.append({
                            "id": safe_id,
                            "values": embeddings[i].tolist(),
                            "metadata": {"text": chunk.page_content, "source": chunk.metadata['source']}
                        })
                    
                    index.upsert(vectors=vectors)
                    st.success(f"✅ Ingestion Complete! KB updated with {len(vectors)} new vectors.")
                    st.session_state.vectors_uploaded = True

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Restore specific UI elements for assistant messages
        if msg.get("metrics"):
            st.markdown(msg["metrics"], unsafe_allow_html=True)
        
        if msg.get("sources"):
            with st.expander("🔍 View Referenced Sources"):
                for source in msg["sources"]:
                    st.markdown(source["label"])
                    st.info(source["text"])
                    st.caption(source["caption"])

if query := st.chat_input("Ask a question about your document..."):
    # 1. Handle User Input
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Handle AI Response
    with st.chat_message("assistant"):
        status = st.empty()
        status.caption("🔍 Phase 1: Vector Search (Pinecone)...")
        
        try:
            # Pipeline Logic
            model = get_embedding_model()
            query_vec = model.encode(query).tolist()
            
            index = get_pinecone_index()
            results = index.query(vector=query_vec, top_k=20, include_metadata=True)
            retrieved_texts = [match['metadata']['text'] for match in results['matches']]
            
            status.caption("⚖️ Phase 2: Neural Reranking (Cohere)...")
            co = get_cohere_client()
            rerank_results = co.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=retrieved_texts,
                top_n=5
            )
            
            final_context = [retrieved_texts[hit.index] for hit in rerank_results.results]
            
            status.caption("🧠 Phase 3: Generation (Gemini)...")
            start_time = time.time()
            client = configure_gemini()
            citation_context = format_docs_for_prompt(final_context)
            
            prompt = f"""
You are a professional Retrieval-Augmented Generation (RAG) assistant. 
Your goal is to answer the user's question accurately using ONLY the provided context blocks.

### INSTRUCTIONS:
1. USE CONTEXT: Base your answer solely on the provided context snippets labeled [1], [2], [3], etc.
2. CITATIONS: Every claim you make MUST be followed by an inline citation. 
   - Example: "The company's revenue grew by 20% in 2023 [1]."
   - If a sentence uses information from multiple sources, use [1][2].
3. FALLBACK: If the provided context does not contain enough information to answer the question, state: 
   "I'm sorry, but the provided documents do not contain the information needed to answer this question." 
   EXCEPTION: 
    - If the user asks general questions ("summarize", "what is this"), SUMMARIZE the content.
    - If the user greets you ("hi", "hello"), answer politely and ask how you can help with the document.
   Do NOT use outside knowledge for specific facts.
4. FORMATTING: Use clear, concise language and bullet points if the answer is complex.

### PROVIDED CONTEXT:
{citation_context}

### USER QUESTION:
{query}
"""
            response = client.models.generate_content(model='gemini-2.0-flash', contents=prompt)
            end_time = time.time()
            
            # Metrics
            duration = end_time - start_time
            import tiktoken
            encoder = tiktoken.encoding_for_model("gpt-4")
            input_tokens = len(encoder.encode(prompt))
            output_tokens = len(encoder.encode(response.text))
            total_tokens = input_tokens + output_tokens
            cost = (input_tokens * 0.075 / 1_000_000) + (output_tokens * 0.30 / 1_000_000)

            status.empty()
            
            # Display & Save
            header_text = "### ✨ AI Analysis"
            st.markdown(header_text)
            st.markdown(response.text)
            
            metrics_html = f"""
            <div style="
                text-align: right;
                padding: 10px 0;
                color: #666;
                font-size: 0.85em;
                font-family: sans-serif;
                border-top: 1px solid #eee;
                margin-top: 10px;
            ">
                <span>⏱️ {duration:.2f}s</span> &nbsp;|&nbsp; 
                <span>⚡ {total_tokens} tokens</span> &nbsp;|&nbsp; 
                <span>🪙 ${cost:.6f}</span>
            </div>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            source_data = []
            with st.expander("🔍 View Referenced Sources"):
                for i, text in enumerate(final_context):
                    label = f"**Source [{i+1}]**"
                    caption = f"Relevance Score: N/A (Top {i+1} match)"
                    st.markdown(label)
                    st.info(text)
                    st.caption(caption)
                    source_data.append({"label": label, "text": text, "caption": caption})
            
            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"{header_text}\n\n{response.text}",
                "metrics": metrics_html,
                "sources": source_data
            })
            
        except Exception as e:
            status.error(f"Error: {e}")
