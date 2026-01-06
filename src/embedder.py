"""
Embedder - Vektör Oluşturma Modülü

Bu modül, metin parçalarını sayısal vektörlere dönüştürür.

RAG Sistemindeki Rolü:
- Metinleri matematiksel temsillere çevirir
- Benzer metinler → Yakın vektörler
- Semantic (anlamsal) arama mümkün olur

Embedding Nasıl Çalışır?
- Her kelime/cümle bir vektör olur
- Örn: 384 boyutlu bir vektör [0.12, -0.45, 0.78, ...]
- "Köpek" ve "Kedi" vektörleri birbirine yakın
- "Köpek" ve "Araba" vektörleri birbirinden uzak

Kullanılan Model: sentence-transformers/all-MiniLM-L6-v2
- 384 boyutlu embedding
- Hızlı ve hafif
- Çok dilli destek
"""

from typing import List, Union
import numpy as np


class Embedder:
    """
    Metin Embedding Oluşturucu

    Sentence Transformers kütüphanesini kullanarak
    metinleri dense vektörlere dönüştürür.

    Örnek kullanım:
    >>> embedder = Embedder()
    >>> vector = embedder.embed_text("Merhaba dünya")
    >>> print(f"Vektör boyutu: {len(vector)}")
    Vektör boyutu: 384
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            model_name: Kullanılacak embedding modeli
                        Varsayılan model hızlı ve etkilidir.
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Embedding modelini yükler."""
        try:
            from sentence_transformers import SentenceTransformer

            print(f"🔄 Embedding modeli yükleniyor: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Model bilgilerini göster
            embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model yüklendi! Embedding boyutu: {embedding_dim}")

        except ImportError:
            raise ImportError(
                "sentence-transformers paketi gerekli!\n"
                "Kurulum: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> List[float]:
        """
        Tek bir metni vektöre dönüştürür.

        Args:
            text: Embed edilecek metin

        Returns:
            Float listesi (embedding vektörü)
        """
        if not text or not text.strip():
            raise ValueError("Boş metin embed edilemez!")

        # Model ile embedding oluştur
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Birden fazla metni vektörlere dönüştürür.

        Args:
            texts: Metin listesi
            batch_size: Batch boyutu (bellek optimizasyonu için)

        Returns:
            Embedding vektörlerinin listesi
        """
        if not texts:
            return []

        # Boş metinleri filtrele
        valid_texts = [t for t in texts if t and t.strip()]

        if not valid_texts:
            return []

        print(f"🔄 {len(valid_texts)} metin embed ediliyor...")

        # Batch halinde embed et
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=len(valid_texts) > 10,
            convert_to_numpy=True,
        )

        print(f"✅ Embedding tamamlandı!")

        return embeddings.tolist()

    def embed_chunks(self, chunks: List) -> List[dict]:
        """
        Chunk nesnelerini embed eder.

        Args:
            chunks: Chunk nesnelerinin listesi

        Returns:
            Her chunk için {chunk, embedding} sözlüğü listesi
        """
        # Chunk içeriklerini çıkar
        texts = [chunk.content for chunk in chunks]

        # Embed et
        embeddings = self.embed_texts(texts)

        # Chunk-embedding eşleştirmesi
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({"chunk": chunk, "embedding": embedding})

        return results

    def get_embedding_dimension(self) -> int:
        """Embedding vektörünün boyutunu döndürür."""
        return self.model.get_sentence_embedding_dimension()

    def compute_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """
        İki embedding arasındaki kosinüs benzerliğini hesaplar.

        Kosinüs Benzerliği:
        - 1.0: Tamamen aynı yönde (çok benzer)
        - 0.0: Dik (ilişkisiz)
        - -1.0: Zıt yönde (zıt anlam)

        Args:
            embedding1: İlk vektör
            embedding2: İkinci vektör

        Returns:
            Benzerlik skoru (-1 ile 1 arası)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Kosinüs benzerliği: (a · b) / (||a|| * ||b||)
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


def demonstrate_embeddings():
    """Embedding kavramını görselleştirir."""

    embedder = Embedder()

    # Test cümleleri
    sentences = [
        "Python programlama dili",
        "Python yazılım geliştirme",
        "Java programlama dili",
        "Bugün hava çok güzel",
        "Kediler sevimli hayvanlardır",
    ]

    print("\n" + "=" * 60)
    print("EMBEDDING DEMONSTRASYONu")
    print("=" * 60)

    # Embedding'leri oluştur
    embeddings = embedder.embed_texts(sentences)

    # Benzerlik matrisi
    print("\n📊 Cümle Benzerlik Matrisi:")
    print("-" * 60)

    # Başlık satırı
    header = "         "
    for i in range(len(sentences)):
        header += f"  S{i+1}  "
    print(header)

    for i, sent1 in enumerate(sentences):
        row = f"S{i+1}      "
        for j, sent2 in enumerate(sentences):
            similarity = embedder.compute_similarity(embeddings[i], embeddings[j])
            row += f" {similarity:.2f} "
        print(row)
        print(f"   → {sent1[:40]}...")

    print("\n💡 Yorum:")
    print("- S1 ve S2 (Python konusu) yüksek benzerlik gösterir")
    print("- S1 ve S3 (programlama dilleri) orta benzerlik")
    print("- S1 ve S4/S5 (farklı konular) düşük benzerlik")


# Test için
if __name__ == "__main__":
    demonstrate_embeddings()
