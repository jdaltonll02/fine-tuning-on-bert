import os
import math
import torch
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class CulinaryRAG:
    def __init__(self, model_path="./models/faiss_index",
                 embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embed_model_name = embed_model
        self.model_path = model_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
        )
        self.vectorstore = None
        self.retriever = None

    def load_index(self):
        """Load FAISS index from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError("❌ FAISS index not found. Build it first.")

        self.vectorstore = FAISS.load_local(
            self.model_path, self.embeddings, allow_dangerous_deserialization=True
        )
        self.retriever = self.vectorstore.as_retriever()
        return self.retriever

    def build_index(self, docs, batch_size=5000):
        """Build a new FAISS index in batches to avoid OOM and save it."""
        total_batches = math.ceil(len(docs) / batch_size)
        print(f"Building FAISS index in {total_batches} batches...")

        # Build first batch
        first_batch = docs[:batch_size]
        faiss_index = FAISS.from_documents(first_batch, self.embeddings)

        # Add remaining batches
        for i in tqdm(range(batch_size, len(docs), batch_size), desc="Batches"):
            batch_docs = docs[i:i+batch_size]
            batch_vectors = [self.embeddings.embed_query(d.page_content) for d in batch_docs]

            # Add to FAISS index
            faiss_index.index.add(batch_vectors)

            # Update docstore mappings
            for idx, doc in enumerate(batch_docs):
                doc_id = str(len(faiss_index.docstore) + idx)
                faiss_index.docstore.add_text(doc_id, doc)
                faiss_index.index_to_docstore_id[faiss_index.index.ntotal - len(batch_docs) + idx] = doc_id

        # Assign to object
        self.vectorstore = faiss_index
        self.retriever = self.vectorstore.as_retriever()

        # Save to disk
        os.makedirs(self.model_path, exist_ok=True)
        self.vectorstore.save_local(self.model_path)
        print("✅ FAISS index built and saved successfully!")

    def query(self, question: str):
        """Run a retrieval query."""
        if self.retriever is None:
            raise ValueError("Retriever not loaded. Call load_index() first.")

        docs = self.retriever.get_relevant_documents(question)
        return docs
