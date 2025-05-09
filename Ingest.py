import os
from pydoc import doc
import pandas as pd

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from faiss import IndexFlatL2
import logging

# -----------------------------
# Logging Setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# # -----------------------------
# # Environment Setup
# # -----------------------------
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Ensure fallback to CPU on macOS
# device = "cpu"

# -----------------------------
# Step 1: Load and Combine CSV
# -----------------------------
logging.info("Loading CSV data...")
df = pd.read_csv("data/bns_sections.csv")

logging.info("Combining columns to form documents...")
df['combined_text'] = (
    "Chapter: " + df['Chapter'].astype(str) + " - " + df['Chapter_name'].fillna('') + 
    " (" + df['Chapter_subtype'].fillna('') + ")\n" +
    "Section: " + df['Section'].astype(str) + " - " + df['Section_name'].fillna('') + "\n" +
    df['Description'].fillna('')
)
documents = df['combined_text'].head(5).tolist()  # Use only the first 5 documents for debugging

print(documents)

# -----------------------------
# Step 2: Split into Chunks
# -----------------------------
logging.info("Extracting and splitting texts from documents...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
texts = []
for document in documents:
    if hasattr(document, 'get_text'):
        text_content = document.get_text()  # Adjust according to actual method
    else:
        text_content = ""  # Default to empty string if no text method is available

    texts.extend(text_splitter.split_text(text_content))

# -----------------------------
# Step 3: Load Embeddings
# -----------------------------
logging.info("Loading embedding model...")
def embedding_function(text):
    embeddings_model = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
    return embeddings_model.embed_query(text)

# Create FAISS index for embeddings
index = IndexFlatL2(768)  # Dimension of embeddings, adjust as needed
# -----------------------------
# Step 4: Vectorize and Save
# -----------------------------


docstore = {i: text for i, text in enumerate(texts)}
index_to_docstore_id = {i: i for i in range(len(texts))}

# Initialize FAISS
faiss_db = FAISS(embedding_function, index, docstore, index_to_docstore_id)

# Process and store embeddings
logging.info("Storing embeddings in FAISS...")
for i, text in enumerate(texts):
    embedding = embedding_function(text)
    faiss_db.add_documents([embedding])

# Exporting the vector embeddings database with logging
logging.info("Exporting the vector embeddings database...")
faiss_db.save_local("bns_embed_db")

# Log a message to indicate the completion of the process
logging.info("Process completed successfully.")