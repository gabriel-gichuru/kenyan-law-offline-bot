import streamlit as st
import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Configuration
DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "kenya_laws"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "phi3"

SYSTEM_PROMPT = """You are a strict, highly concise Kenyan legal guide. 
1. Answer using ONLY the provided Context. If it does not contain the answer, state: "I cannot answer this based on the provided laws."
2. You MUST use this exact two-sentence format for every response:
[Your direct one-sentence answer]. According to [Document Name], [Section/Article Number], [your one-sentence explanation].
3. DO NOT use labels like "Answer:" or "Explanation:". Write in plain paragraphs.
4. NEVER generate follow-up questions. Stop writing immediately after your explanation.
5. Always respond in simple English."""

# Your Custom Sheng/Swahili to English Dictionary
SHENG_DICTIONARY = {
    "niaje": "hello",
    "kwa ngori": "in trouble",
    "ngori": "trouble",
    "karao": "police officer",
    "polisi": "police officer",
    "kanjo": "county council officer",
    "hongo": "bribe",
    "ndai": "vehicle",
    "moti": "vehicle",
    "mathogothanio": "public nuisance",
    "ashikwe": "arrested",
    "anadai": "he/she is saying/wanting",
    "sumbua": "to harass or annoy",
    "makarao": "police officers",
    "iba": "steal",
    "haribu": "destroy",
    "haribiwa": "is destroyed",
    "chai": "bribe/tea",
    "soda": "soft drink/bribe",
    "wanadai": "they are saying/wanting",
    "dirty car": "obscured identification plates maintenance"
}

def translate_sheng_to_english(text):
    """Replaces Sheng words with English equivalents before processing."""
    translated_text = text.lower()
    for sheng_word, english_word in SHENG_DICTIONARY.items():
        pattern = r'\b' + re.escape(sheng_word) + r'\b'
        translated_text = re.sub(pattern, english_word, translated_text)
    return translated_text

# Initialize ChromaDB Database
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_func)

collection = get_chroma_collection()

# Streamlit UI Setup
st.set_page_config(page_title="Kenyan Law Bot", page_icon="⚖️")
st.title("Offline Kenyan Law Source")
st.markdown("Ask practical questions about the Constitution, Traffic laws, Penal code, and Nairobi by-laws.")

# Clear Chat Button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a legal question..."):
    # Display user message on UI exactly as they typed it
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response with dynamic status and streaming
    with st.chat_message("assistant"):
        
        # Interactive status container showing exact background tasks
        with st.status("Analyzing your question...", expanded=True) as status:
            
            st.write("Translating slang and Swahili...")
            translated_prompt = translate_sheng_to_english(prompt)

            st.write("Searching offline vector database...")
            results = collection.query(
                query_texts=[translated_prompt],
                n_results=7,
                include=["documents", "metadatas"]
            )
            
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []

            context_parts = []
            for doc, meta in zip(documents, metadatas):
                source = meta.get("source_file", "Unknown")
                kind = meta.get("kind", "provision")
                number = meta.get("number", "Unknown")
                title = meta.get("title", "Untitled")
                context_parts.append(f"[{source} | {kind.title()} {number}: {title}]\n{doc}")
            
            context = "\n\n".join(context_parts)

            st.write("Generating legal response...")
            messages_for_ollama = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages_for_ollama.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {translated_prompt}\n\nAnswer:"
            })

            # Start Ollama generation with the stop sequence
            stream = ollama.chat(
                model=OLLAMA_MODEL, 
                messages=messages_for_ollama, 
                stream=True,
                options={"stop": ["Question:", "Context:", "\n\n"]}
            )
            
            status.update(label="Response is being generated...", state="complete", expanded=False)

        # Output the streamed text below the status box
        response_placeholder = st.empty()
        full_response = ""
        
        for chunk in stream:
            full_response += chunk["message"]["content"]
            response_placeholder.markdown(full_response + "▌")
            
        response_placeholder.markdown(full_response)
        
    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})