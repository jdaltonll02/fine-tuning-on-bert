import os
import glob
import math
from tqdm import tqdm
import torch
import numpy as np

from langchain_community.document_loaders import CSVLoader, JSONLoader
from rag_system.rag_pipeline import CulinaryRAG
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_documents(data_folder="./data"):
    """Load CSV and JSON files into LangChain Documents."""
    docs = []

    # Load CSV files
    for csv_file in glob.glob(os.path.join(data_folder, "*.csv")):
        print(f"Loading CSV: {csv_file}")
        loader = CSVLoader(file_path=csv_file)
        docs.extend(loader.load())

    # Load JSON files
    for json_file in glob.glob(os.path.join(data_folder, "*.json")):
        print(f"Loading JSON: {json_file}")
        loader = JSONLoader(file_path=json_file, jq_schema=".[]", text_content=False)
        docs.extend(loader.load())

    print(f"Total documents loaded: {len(docs)}")
    return docs

def build_index_in_batches(docs, rag, batch_size=5000):
    """Build a FAISS index in batches with safe memory usage."""
    embeddings = HuggingFaceEmbeddings(
        model_name=rag.embed_model_name,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    total_batches = math.ceil(len(docs) / batch_size)
    print(f"Building FAISS index in {total_batches} batches...")

    faiss_index = None

    for batch_num, start_idx in enumerate(range(0, len(docs), batch_size), 1):
        batch_docs = docs[start_idx:start_idx + batch_size]
        batch_texts = [d.page_content for d in batch_docs]

        print(f"Processing batch {batch_num} with {len(batch_docs)} documents...")

        # Embed batch
        batch_vectors = embeddings.embed_documents(batch_texts)
        batch_vectors = np.array(batch_vectors).astype("float32")

        # Create FAISS index for this batch
        batch_faiss = FAISS.from_texts(batch_texts, embeddings)

        if faiss_index is None:
            faiss_index = batch_faiss
            print("  -> First batch FAISS index created.")
        else:
            # Merge batch vectors into main FAISS index
            faiss_index.index.add(batch_vectors)
            start_id = len(faiss_index.index_to_docstore_id)
            for idx, doc in enumerate(batch_docs):
                doc_id = str(start_id + idx)
                faiss_index.docstore._dict[doc_id] = doc  # use internal dict for InMemoryDocstore
                faiss_index.index_to_docstore_id[faiss_index.index.ntotal - len(batch_docs) + idx] = doc_id
            print(f"  -> Batch merged into main FAISS index.")

    # Assign to RAG
    rag.vectorstore = faiss_index
    rag.retriever = faiss_index.as_retriever()

    # Save FAISS index
    rag.vectorstore.save_local(rag.model_path)
    print(f"\n✅ FAISS index built and saved successfully at '{rag.model_path}'!")
    print(f"Total documents indexed: {faiss_index.index.ntotal}")

def main():
    docs = load_documents()
    if len(docs) == 0:
        print("No documents found in /data. Exiting.")
        return

    rag = CulinaryRAG()
    build_index_in_batches(docs, rag, batch_size=5000)

if __name__ == "__main__":
    main()
