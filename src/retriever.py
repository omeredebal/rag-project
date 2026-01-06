"""
Retriever - Bilgi Getirme Modülü

Bu modül, kullanıcı sorgusuna en uygun chunk'ları getirir.

RAG Sistemindeki Rolü:
- Kullanıcı sorusunu anlama
- En alakalı bilgileri bulma
- LLM'e context sağlama

Retrieval Süreci:
1. Sorgu → Embedding
2. Embedding → Vector Store'da arama
3. Top-K en benzer chunk'ları getir
4. Sonuçları sırala ve döndür

Neden Retrieval Önemli?
- LLM'in bilgi kapasitesi sınırlı
- Güncel/özel bilgiler LLM'de yok
- Retrieval ile doğru context → Doğru yanıt
"""

from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Retrieval sonucunu temsil eden veri sınıfı"""

    content: str  # Chunk içeriği
    score: float  # Benzerlik skoru
    metadata: Dict  # Kaynak bilgisi

    def __repr__(self):
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return f"RetrievalResult(score={self.score:.3f}, content='{preview}')"


class Retriever:
    """
    Bilgi Getirici

    Kullanıcı sorgusuna göre en alakalı chunk'ları
    vector store'dan getirir.

    Örnek kullanım:
    >>> retriever = Retriever(embedder, vector_store)
    >>> results = retriever.retrieve("Python nedir?", top_k=3)
    >>> for r in results:
    >>>     print(f"Score: {r.score:.2f} - {r.content[:50]}...")
    """

    def __init__(
        self,
        embedder,  # Embedder instance
        vector_store,  # VectorStore instance
        top_k: int = 5,
        score_threshold: float = 0.0,
    ):
        """
        Args:
            embedder: Embedding oluşturucu
            vector_store: Vektör deposu
            top_k: Varsayılan sonuç sayısı
            score_threshold: Minimum skor eşiği (altındakiler filtrelenir)
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Sorguya en uygun chunk'ları getirir.

        Args:
            query: Kullanıcı sorusu
            top_k: Döndürülecek sonuç sayısı
            filter_metadata: Metadata filtresi

        Returns:
            RetrievalResult listesi (skora göre sıralı)
        """
        if not query or not query.strip():
            print("⚠️  Boş sorgu!")
            return []

        k = top_k or self.top_k

        print(f"🔍 Aranıyor: '{query[:50]}...' (top_k={k})")

        # 1. Sorguyu embed et
        query_embedding = self.embedder.embed_text(query)

        # 2. Vector store'da ara
        search_results = self.vector_store.search(
            query_embedding=query_embedding, top_k=k, where=filter_metadata
        )

        # 3. Sonuçları dönüştür ve filtrele
        results = []
        for sr in search_results:
            print(f"   📊 Skor: {sr.score:.3f} (eşik: {self.score_threshold})")
            # Skor eşiğini kontrol et
            if sr.score >= self.score_threshold:
                results.append(
                    RetrievalResult(
                        content=sr.content, score=sr.score, metadata=sr.metadata
                    )
                )

        print(
            f"✅ {len(results)} sonuç bulundu (filtrelendi: {len(search_results) - len(results)})"
        )

        return results

    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_separator: str = "\n\n---\n\n",
    ) -> str:
        """
        Sonuçları birleştirilmiş context olarak döndürür.

        Bu metod, LLM'e verilecek context string'ini oluşturur.

        Args:
            query: Kullanıcı sorusu
            top_k: Sonuç sayısı
            context_separator: Chunk'lar arası ayırıcı

        Returns:
            Birleştirilmiş context string
        """
        results = self.retrieve(query, top_k)

        if not results:
            return ""

        # Chunk'ları birleştir
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("filename", "Bilinmeyen")
            context_parts.append(f"[Kaynak {i}: {source}]\n{result.content}")

        return context_separator.join(context_parts)

    def get_sources(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Sonuçların kaynaklarını döndürür.

        Kullanıcıya "Bu bilgi şu kaynaklardan geldi" demek için.
        """
        results = self.retrieve(query, top_k)

        sources = []
        for r in results:
            source = r.metadata.get("source", r.metadata.get("filename", "Bilinmeyen"))
            if source not in sources:
                sources.append(source)

        return sources


class HybridRetriever:
    """
    Hibrit Retriever (İleri Seviye)

    Semantic search + Keyword search birleştirir.
    Bu, basit projede opsiyoneldir.
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve(
        self, query: str, top_k: int = 5, keyword_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Semantic ve keyword aramayı birleştirir.

        Not: Bu basitleştirilmiş bir implementasyon.
        Gerçek hibrit arama için BM25 + dense retrieval kullanılır.
        """
        # Şimdilik sadece semantic arama
        return self.retriever.retrieve(query, top_k)


# Test için
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RETRIEVER TEST")
    print("=" * 60)
    print("Not: Bu test gerçek Embedder ve VectorStore gerektirir.")
    print("Demo.py dosyasını çalıştırarak tam testi yapabilirsiniz.")
