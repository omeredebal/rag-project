"""
Vector Store - Vektör Depolama Modülü

Bu modül, embedding vektörlerini depolar ve benzerlik araması yapar.

RAG Sistemindeki Rolü:
- Chunk embedding'lerini kalıcı olarak saklar
- Hızlı benzerlik araması sağlar
- Metadata ile filtreleme imkanı sunar

Kullanılan Teknoloji: ChromaDB
- Açık kaynak vektör veritabanı
- Lokal çalışır (sunucu gerektirmez)
- Kalıcı depolama desteği
- Metadata filtreleme

Benzerlik Araması Nasıl Çalışır?
1. Sorgu metni embed edilir → sorgu vektörü
2. Tüm chunk vektörleri ile mesafe hesaplanır
3. En yakın K vektör döndürülür
"""

import os
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Arama sonucunu temsil eden veri sınıfı"""

    content: str  # Chunk içeriği
    metadata: Dict  # Chunk metadata'sı
    distance: float  # Sorguya olan uzaklık (düşük = daha benzer)
    score: float  # Benzerlik skoru (yüksek = daha benzer)

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"SearchResult(score={self.score:.3f}, preview='{preview}')"


class VectorStore:
    """
    Vektör Deposu (ChromaDB Wrapper)

    Embedding vektörlerini depolar ve benzerlik araması yapar.

    Örnek kullanım:
    >>> store = VectorStore(collection_name="my_docs")
    >>> store.add_documents(chunks, embeddings)
    >>> results = store.search(query_embedding, top_k=5)
    """

    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
    ):
        """
        Args:
            collection_name: Koleksiyon adı (grup/tablo benzeri)
            persist_directory: Veritabanı kayıt dizini
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None

        self._initialize_db()

    def _initialize_db(self):
        """ChromaDB'yi başlatır."""
        try:
            import chromadb

            print(f"🔄 ChromaDB başlatılıyor...")

            # Kalıcı depolama ile client oluştur
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
            )

            # Koleksiyonu al veya oluştur
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG document embeddings"},
            )

            doc_count = self.collection.count()
            print(
                f"✅ Koleksiyon hazır: '{self.collection_name}' ({doc_count} döküman)"
            )

        except ImportError:
            raise ImportError(
                "chromadb paketi gerekli!\n" "Kurulum: pip install chromadb"
            )

    def add_documents(
        self,
        chunks: List,
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """
        Chunk'ları ve embedding'lerini depoya ekler.

        Args:
            chunks: Chunk nesnelerinin listesi
            embeddings: Embedding vektörlerinin listesi
            ids: Benzersiz ID'ler (opsiyonel, otomatik oluşturulur)

        Returns:
            Eklenen döküman sayısı
        """
        if not chunks or not embeddings:
            print("⚠️  Eklenecek döküman yok!")
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError("Chunk ve embedding sayıları eşleşmiyor!")

        # ID'leri hazırla
        if ids is None:
            ids = [f"doc_{i}_{hash(chunk.content)}" for i, chunk in enumerate(chunks)]

        # İçerikleri ve metadata'ları ayır
        documents = [chunk.content for chunk in chunks]
        metadatas = []

        for chunk in chunks:
            # Metadata'yı ChromaDB formatına çevir
            meta = {}
            if hasattr(chunk, "metadata") and chunk.metadata:
                for key, value in chunk.metadata.items():
                    # ChromaDB sadece string, int, float, bool kabul eder
                    if isinstance(value, (str, int, float, bool)):
                        meta[key] = value
                    else:
                        meta[key] = str(value)
            metadatas.append(meta)

        # ChromaDB'ye ekle
        print(f"🔄 {len(chunks)} chunk ekleniyor...")

        self.collection.add(
            ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
        )

        print(f"✅ {len(chunks)} chunk başarıyla eklendi!")
        return len(chunks)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Benzerlik araması yapar.

        Args:
            query_embedding: Sorgu vektörü
            top_k: Döndürülecek sonuç sayısı
            where: Metadata filtresi (örn: {"source": "doc1.txt"})
            where_document: İçerik filtresi (örn: {"$contains": "python"})

        Returns:
            SearchResult listesi (skora göre sıralı)
        """
        # Sorguyu çalıştır
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        # Sonuçları SearchResult nesnelerine dönüştür
        search_results = []

        if results and results["documents"] and results["documents"][0]:
            documents = results["documents"][0]
            metadatas = (
                results["metadatas"][0]
                if results["metadatas"]
                else [{}] * len(documents)
            )
            distances = (
                results["distances"][0]
                if results["distances"]
                else [0] * len(documents)
            )

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Distance'ı score'a çevir (1 / (1 + distance))
                # Düşük distance = yüksek score
                score = 1 / (1 + dist)

                search_results.append(
                    SearchResult(
                        content=doc, metadata=meta or {}, distance=dist, score=score
                    )
                )

        return search_results

    def delete_collection(self):
        """Koleksiyonu tamamen siler."""
        if self.client and self.collection_name:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️  Koleksiyon silindi: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        """Koleksiyon istatistiklerini döndürür."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": self.persist_directory,
        }

    def clear(self):
        """Tüm dökümanları siler ama koleksiyonu korur."""
        # Mevcut koleksiyonu sil ve yeniden oluştur
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG document embeddings"},
        )
        print(f"🧹 Koleksiyon temizlendi: {self.collection_name}")


# Test için
if __name__ == "__main__":
    import tempfile

    # Geçici dizinde test
    with tempfile.TemporaryDirectory() as tmpdir:
        print("\n" + "=" * 60)
        print("VECTOR STORE TEST")
        print("=" * 60)

        # Store oluştur
        store = VectorStore(collection_name="test_collection", persist_directory=tmpdir)

        # Test verisi (basit mock chunk ve embedding)
        class MockChunk:
            def __init__(self, content, metadata=None):
                self.content = content
                self.metadata = metadata or {}

        chunks = [
            MockChunk("Python bir programlama dilidir", {"source": "doc1.txt"}),
            MockChunk("Machine learning yapay zeka dalıdır", {"source": "doc2.txt"}),
            MockChunk(
                "Web geliştirme frontend ve backend içerir", {"source": "doc3.txt"}
            ),
        ]

        # Basit rastgele embedding (gerçek projede Embedder kullanılır)
        import random

        embeddings = [[random.random() for _ in range(384)] for _ in chunks]

        # Ekle
        store.add_documents(chunks, embeddings)

        # Ara
        query_embedding = embeddings[0]  # İlk dökümanın embedding'i ile ara
        results = store.search(query_embedding, top_k=2)

        print("\n🔍 Arama Sonuçları:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f} - {result.content[:50]}")

        # İstatistikler
        print(f"\n📊 İstatistikler: {store.get_stats()}")
