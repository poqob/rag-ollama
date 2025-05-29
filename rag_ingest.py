import os
import glob
import sys
import pdfplumber
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF'leri oku ve metinleri birleştir
def load_pdfs_from_docs(docs_path="docs", file_names=None):
    all_text = []
    if file_names:
        pdf_files = [os.path.join(docs_path, f) for f in file_names]
    else:
        pdf_files = glob.glob(os.path.join(docs_path, "*.pdf"))
    for pdf_file in pdf_files:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            all_text.append(text)
    return "\n".join(all_text)

# Metni parçalara ayır
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Embedding modeli
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Komut satırından dosya isimleri al
if __name__ == "__main__":
    file_names = sys.argv[1:] if len(sys.argv) > 1 else None
    raw_text = load_pdfs_from_docs(file_names=file_names)
    texts = text_splitter.split_text(raw_text)
    vectorstore = FAISS.from_texts(texts, embeddings)
    vectorstore.save_local("faiss_index")
    print("Vektör veritabanı oluşturuldu ve kaydedildi.")
