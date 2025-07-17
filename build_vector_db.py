"""
Vector Database Builder
Faiss is a library for efficient similarity search and clustering of dense vectors.
"""

import os
import fitz # PyMuPDF
from sentence_transformers import SentenceTransformer # Sentece Transformers embedding için
import faiss # Faiss kütüphanesi
import numpy as np
import pickle # Veri tabanı dosyalarını kaydetmek için

# program için .pdf yükle
def extract_text_from_pdf(pdf_path):
    """
    PDF Dosyasından metin çıkarır.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text

#print(extract_text_from_pdf(".\data\Consulting_Agreement.pdf"))

# uzun metni daha küçük parçalara ayır gerçek hayatta paragraflar halinde veya başlıklar ve subtitle ile bölüyoruz ve bağlamsal açıdan olanları ayırıp başka yerde birleştiriyoruz

def chunk_text(text, max_length=500):
    """
    Metni belirtieln karakter uzunluğuna göre böl
    """
    chunks = []
    current = ""
    for line in text.split('\n'):
        if len(current) + len(line) < max_length:
            current += " " + line.strip()
        else:
            chunks.append(current.strip())
            current = line.strip()
    if current:
        chunks.append(current.strip())
    return chunks

#text_dummy = extract_text_from_pdf(".\data\Consulting_Agreement.pdf")
#print(chunk_text(text_dummy, max_length=500))

# sentence transformers ile embedding yapalım
# huggingface.co/models adresinden embedding modellerine bakabilirsiniz
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# Bu model, metinleri 384 boyutlu vektörlere dönüştürür ve genellikle iyi bir denge sağlar.
model = SentenceTransformer('all-MiniLM-L6-v2')

# pdf yolunu pelirt 
pdf_file_path = ".\data\Consulting_Agreement.pdf"

# metni çıkar
text = extract_text_from_pdf(pdf_file_path)

# metni chunklara ayır
chunks = chunk_text(text, max_length=500)

# her chunk için embedding yap
embeddings = model.encode(chunks)

print(f"Embedding Shape: {embeddings.shape}") # (Örnek: (n_chunks, 384) boyutunda bir matris)

# faiss  index oluştur
dimension = embeddings.shape[1]  # embedding boyutu = 384
index = faiss.IndexFlatL2(dimension)  # L2 mesafesi kullanarak düz bir indeks oluştur
index.add(np.array(embeddings))  # embeddingleri indekse ekle

#faiss indeksi ve chunkları kaydet
faiss.write_index(index, "./data/contract_index.faiss")
with open("./data/contract_chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("FAISS index and chunks saved successfully.")
