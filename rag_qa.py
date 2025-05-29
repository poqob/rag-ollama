import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# FAISS vektör veritabanını yükle
vectorstore = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Retriever oluştur
retriever = vectorstore.as_retriever()

# Yerel LLM (Ollama mistral:7b)
llm = Ollama(model="mistral:7b")

# RetrievalQA zinciri
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Soru al
if __name__ == "__main__":
    question = input("Soru: ")
    answer = qa.run(question)
    print("Yanıt:", answer)
