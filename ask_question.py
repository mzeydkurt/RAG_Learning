"""
Problem Tanımı : Sözleşme Asistanı(ENG) 
    -kullanıcının yüklediği bir sözleişme dosyasından içerik çıkarmak
    -Bu içeriği vektorel olarak temsil edelim(embedding) yapalım
    - faiss kullanatak hızlı arama yapabilen bir vektör veri tabanı oluştur
    - kullanıcın soruları anlayıp git db den bilgileri getir llm il ecevap üreticez


Kullanılan Teknolojiler:
    - embedding : metni vektörleştirme işlemi
    - fais : vektör veri tabanı hızlı benzerlik aramsı için vektör veri tabanı
    -gpt3.5 metin üretimi ve cevaplama

RAG: Retrieval Augmented Genaraiton: dil modellerine bilgi desteği sağkayan bir teknik
    - kullanıcı sorularını al. ilgili bilgiyi vektör veri tabanından getir, sonra gpt ile cevap üret
    - kullanıcın sorusu embedding e dönüştürülür, faiss üzerinden en alakı içerik(chunk) bulunur
    - augmentation : zenginleştirme, kullanıcın sorusu ve ilgili içerik birleştirilir
    - genaration : gpt ile cevap üretimi

Plan/Program:
    - sözleşme belgesinin hazırlanması, yüklenmesi
    - metin çıkarma ve parçalama
    - embedding ve faiss ile vektör veri tabanı oluşturma
    - soru cevaplama sitemi



Install Libraries: freeze

import libraries

"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai

# .env dosyasından ortam değişkenlerini yükle
load_dotenv()

# Google Gemini API anahtarını al
api_key = os.getenv("GEMINI_API_KEY")

# Google Gemini yapılandırması
genai.configure(api_key=api_key)

# Gemini modelini başlat
model = genai.GenerativeModel("gemini-1.5-flash")

# Embedding modeli (SentenceTransformer)
model_embed = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index dosyasını yükle
index = faiss.read_index("./data/contract_index.faiss")

# Chunklanmış metinleri yükle
with open("./data/contract_chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Soru-cevap döngüsü
while True:
    print("\nÇıkmak için 'exit' yazabilirsiniz.")
    question = input("\nSorunuzu Giriniz (ENG): ")
    
    if question.lower() == "exit":
        print("Çıkılıyor...")
        break

    # Soruyu vektöre dönüştür
    question_embedding = model_embed.encode([question])

    # FAISS ile en yakın 3 chunk'ı bul
    k = 3
    distances, indices = index.search(np.array(question_embedding), k)

    # Chunk'ları al ve birleştir
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n------\n".join(retrieved_chunks)

    prompt = f"""You are a contract assistant. Based on the contract context below, 
            answer the user's question clearly.

            Context: {context}

            Question: {question}

            Answer:
            """



    # Gemini modeliyle yanıt oluştur
    response = model.generate_content(prompt)

    # Yanıtı yazdır
    print("\nAI Asistan Cevap:\n", response.text.strip())
