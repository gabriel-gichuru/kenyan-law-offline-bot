from pathlib import Path
import re

import chromadb
import fitz  # PyMuPDF
from chromadb.utils import embedding_functions

DB_DIR = str(Path("chroma_db"))


def extract_clean_text(pdf_path: Path) -> str:
    full_text = []
    header_patterns = (
        re.compile(r"^Constitution of Kenya,\s*2010$", re.IGNORECASE),
        re.compile(
            r"^Nairobi City County Public Nuisance(?:\s+Act)?$",
            re.IGNORECASE,
        ),
        re.compile(r"^No\.?\s*\d+$", re.IGNORECASE),
        re.compile(r"^(?:19|20)\d{2}$"),
        re.compile(r"^\d+$"),
    )

    with fitz.open(pdf_path) as doc:
        for page in doc:
            cleaned_lines = []
            for line in page.get_text("text").splitlines():
                line_str = line.strip()
                if any(pattern.fullmatch(line_str) for pattern in header_patterns):
                    continue
                cleaned_lines.append(line)
            full_text.append("\n".join(cleaned_lines))

    return "\n".join(full_text)


def chunk_legal_text(text: str, source_file: str) -> list[dict]:
    section_pattern = re.compile(r"(?m)^\s*(\d{1,3})\.\s*(.*)$")
    matches = list(section_pattern.finditer(text))
    chunks = []
    kind = "article" if "constitution" in source_file.casefold() else "section"

    for i, match in enumerate(matches):
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        number = match.group(1)
        first_line = match.group(2).strip()
        previous_lines = text[:match.start()].splitlines()
        previous_line = next(
            (line.strip() for line in reversed(previous_lines) if line.strip()),
            "",
        )

        is_subsection_start = first_line.startswith("(")
        clause_starter = re.match(
            r"^(?:In|The|A|An|For|Subject|Notwithstanding|Where|Every|Any|No|"
            r"Despite|Upon|Unless|If|Except)\b",
            first_line,
            re.IGNORECASE,
        )
        previous_line_is_title = bool(
            re.fullmatch(
                r"[A-Z][A-Za-z0-9,\s\-'\(\)]{0,118}\.?",
                previous_line,
            )
        )
        use_previous_title = previous_line_is_title and (
            is_subsection_start or not first_line or bool(clause_starter)
        )
        title = previous_line if use_previous_title else first_line
        if not title:
            title = "General Provision"

        body = text[match.start():end_idx].strip()
        if use_previous_title:
            body = f"{title}\n{body}"

        chunks.append({
            "id": f"{source_file}_{kind}_{number}_{i}",
            "text": body,
            "metadata": {
                "number": number,
                "kind": kind,
                "title": title,
                "source_file": source_file,
            }
        })

    return chunks


def store_in_chromadb(chunks: list[dict]) -> None:
    print(f"Initializing local persistent ChromaDB at {DB_DIR}...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Initialize the lightweight local sentence-transformers model
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    collection = client.get_or_create_collection(
        name="kenya_laws",
        embedding_function=embedding_func
    )

    print(f"Embedding and storing {len(chunks)} chunks into 'kenya_laws'...")

    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch]
        )
        print(f"Stored chunks {i + 1} to {min(i + batch_size, len(chunks))}")

    print("\nPhase 1 Complete: Vector store successfully generated!")


if __name__ == "__main__":
    all_chunks = []
    
    for pdf_path in sorted(Path.cwd().glob("*.pdf")):
        raw_text = extract_clean_text(pdf_path)
        chunks = chunk_legal_text(raw_text, pdf_path.name)
        print(f"Extracted {len(chunks)} chunks from {pdf_path.name}.")
        all_chunks.extend(chunks)
        
    for txt_path in sorted(Path.cwd().glob("*.txt")):
        raw_text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_legal_text(raw_text, txt_path.name)
        print(f"Extracted {len(chunks)} chunks from {txt_path.name}.")
        all_chunks.extend(chunks)

    store_in_chromadb(all_chunks)