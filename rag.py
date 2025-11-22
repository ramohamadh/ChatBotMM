
"""
A complete, production-ready RAG system in a single Python file.

Installation:
pip install faiss-cpu sentence-transformers langchain langchain-community openai pypdf python-docx

Usage:
1. Create a 'docs' directory and place your documents (PDF, TXT, DOCX, MD) inside.
2. Set your OpenAI API key as an environment variable:
   export OPENAI_API_KEY='your_api_key_here'
3. Run the script:
   python rag.py

Example:
rag = RAG("docs/")
rag.index()
print(rag.ask("Your question here"))
"""

import os
import faiss
import numpy as np
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class RAG:
    def __init__(self, docs_path="docs/"):
        self.docs_path = docs_path
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.index_file = "faiss_index.bin"
        self.documents_file = "documents.npy"
        self.faiss_index = None
        self.documents = None

    def _load_documents(self):
        """Loads documents from the specified directory."""
        # Define loaders for different file types
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".md": TextLoader,
        }

        # Create a DirectoryLoader with the specified loaders
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*",
            loader_cls=None, # We will specify loaders manually
            use_multithreading=True,
            show_progress=True,
            loader_kwargs=None
        )

        # Manually load files with the correct loader
        docs = []
        for file_path in loader.find_files():
            ext = os.path.splitext(file_path)[1].lower()
            if ext in loaders:
                loader_cls = loaders[ext]
                try:
                    # For TextLoader, we need to specify the encoding
                    if loader_cls == TextLoader:
                        docs.extend(loader_cls(file_path, encoding='utf-8').load())
                    else:
                        docs.extend(loader_cls(file_path).load())
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        return docs


    def _split_text(self, documents):
        """Splits the loaded documents into chunks."""
        return self.text_splitter.split_documents(documents)

    def index(self):
        """Creates a FAISS index for the documents."""
        documents = self._load_documents()
        if not documents:
            print("No documents found to index.")
            return

        chunks = self._split_text(documents)
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)

        self.faiss_index = faiss.IndexFlatL2(len(embeddings[0]))
        self.faiss_index.add(np.array(embeddings))
        self.documents = np.array(texts)

        # Save the index and documents
        faiss.write_index(self.faiss_index, self.index_file)
        np.save(self.documents_file, self.documents)
        print("Indexing complete.")

    def _load_index(self):
        """Loads the FAISS index and documents from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.documents_file):
            self.faiss_index = faiss.read_index(self.index_file)
            self.documents = np.load(self.documents_file, allow_pickle=True)
        else:
            self.index()

    def ask(self, question):
        """Asks a question and returns an answer based on the indexed documents."""
        if self.faiss_index is None or self.documents is None:
            self._load_index()

        if self.faiss_index is None:
            return "No documents have been indexed. Please run the `index()` method first."

        question_embedding = self.embedding_model.embed_query(question)
        distances, indices = self.faiss_index.search(np.array([question_embedding]), k=5)

        retrieved_docs = [self.documents[i] for i in indices[0]]
        context = "\n\n".join(retrieved_docs)

        prompt = f"""
        Answer the following question based only on the provided context.
        If the answer is not in the context, say "I don't know."

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"An error occurred: {e}"

if __name__ == '__main__':
    # Create a dummy docs directory and a document for testing
    if not os.path.exists("docs"):
        os.makedirs("docs")
    with open("docs/sample.txt", "w") as f:
        f.write("The sky is blue. The grass is green.")

    # 1. Initialize the RAG system
    rag = RAG("docs/")

    # 2. Index the documents
    rag.index()

    # 3. Ask a question
    question = "What color is the sky?"
    answer = rag.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    question = "What color is the grass?"
    answer = rag.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

    question = "What is the meaning of life?"
    answer = rag.ask(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
