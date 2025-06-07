import sys
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# FAISS vektör veritabanını yükle
vectorstore = FAISS.load_local("faiss_index", HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

# Retriever oluştur
retriever = vectorstore.as_retriever()

# OllamaLLM'yi streaming ile oluştur
llm = OllamaLLM(
    model="llama3.2:latest",  # veya kullandığınız model
    temperature=0.2,
    callbacks=[StreamingStdOutCallbackHandler()],  # Streaming için callback
    streaming=True  # Streaming özelliğini etkinleştir
)

# RetrievalQA zinciri
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Soru al ve işle
if __name__ == "__main__":
    question = input("Soru: ")
    # Streaming çıktı için invoke kullanıyoruz
    qa.invoke({"query": question})
    print()  # Satır sonu