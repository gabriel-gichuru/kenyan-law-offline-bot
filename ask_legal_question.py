import argparse
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
import ollama


DB_DIR = Path(__file__).resolve().parent / "chroma_db"
COLLECTION_NAME = "kenya_laws"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

SYSTEM_PROMPT = """You are a highly concise Kenyan legal guide. 
Answer the user's question using ONLY the retrieved text. 
You must strictly follow this format:
1. Start with a direct, one-sentence answer.
2. Provide exactly one sentence of explanation for each Article or Section you reference, explicitly naming the Article or Section.
3. Do not include any conversational filler, greetings, or disclaimers about consulting a lawyer.
Respond in simple English or Swahili depending on the user's language."""


def retrieve_chunks(question: str) -> list[tuple[str, dict]]:
    client = chromadb.PersistentClient(path=str(DB_DIR))
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
    )
    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )
    result = collection.query(
        query_texts=[question],
        n_results=3,
        include=["documents", "metadatas"],
    )

    documents = result["documents"]
    metadatas = result["metadatas"]
    if not documents or not documents[0]:
        return []

    retrieved: list[tuple[str, dict]] = []
    for document, metadata in zip(documents[0], metadatas[0]):
        if document:
            retrieved.append((document, metadata or {}))
    return retrieved


def answer_question(question: str) -> str:
    chunks = retrieve_chunks(question)
    if not chunks:
        raise RuntimeError(
            "No legal text was retrieved. Check that the kenya_laws collection "
            "is populated."
        )

    context_parts = []
    for index, (document, metadata) in enumerate(chunks, start=1):
        source_file = metadata.get("source_file", "Unknown source")
        kind = metadata.get("kind", "provision")
        number = metadata.get("number", "Unknown")
        title = metadata.get("title", "Untitled")
        context_parts.append(
            f"[Retrieved passage {index} | {source_file}, "
            f"{kind.title()} {number}: {title}]\n"
            f"{document}"
        )
    context = "\n\n".join(context_parts)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Retrieved legal text:\n\n"
                    f"{context}\n\n"
                    f"User's legal question:\n{question}"
                ),
            },
        ],
    )
    return response["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question about Kenyan law using local RAG."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Legal question (if omitted, you'll be prompted)",
    )
    args = parser.parse_args()
    question = (args.question or input("Ask a Kenyan legal question: ")).strip()
    if not question:
        parser.error("A legal question is required.")

    print(answer_question(question))


if __name__ == "__main__":
    main()
