"""
RAG Pipeline - Ana Sistem Modülü

Bu modül, tüm RAG bileşenlerini birleştiren ana pipeline'dır.

RAG Pipeline Akışı:
1. INDEXING (Bir kez yapılır)
   Dökümanlar → Chunking → Embedding → Vector Store

2. QUERYING (Her soru için)
   Soru → Embedding → Retrieval → LLM → Yanıt

Bu sınıf, tüm adımları orchestrate eder.
"""

import os
from typing import List, Optional, Dict
from dataclasses import dataclass

from .document_loader import DocumentLoader, Document
from .chunker import TextChunker, Chunk
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever, RetrievalResult
from .generator import Generator


@dataclass
class RAGResponse:
    """RAG yanıtını temsil eden veri sınıfı"""

    answer: str  # LLM yanıtı
    sources: List[str]  # Kaynak dökümanlar
    retrieved_chunks: List[RetrievalResult]  # Bulunan chunk'lar

    def __repr__(self):
        return (
            f"RAGResponse(answer='{self.answer[:50]}...', sources={len(self.sources)})"
        )


class RAGPipeline:
    """
    RAG Pipeline

    Tüm RAG bileşenlerini birleştiren ana sınıf.
    End-to-end soru-cevap sistemi sağlar.

    Örnek kullanım:
    >>> rag = RAGPipeline()
    >>> rag.index_documents("data/")
    >>> response = rag.query("Python nedir?")
    >>> print(response.answer)
    """

    def __init__(
        self,
        # Embedding ayarları
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Chunking ayarları
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        # Vector store ayarları
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        # LLM ayarları
        llm_model: str = "llama3.2",
        # Retrieval ayarları
        top_k: int = 3,
    ):
        """
        RAG Pipeline'ı yapılandırır.

        Args:
            embedding_model: Sentence transformer model adı
            chunk_size: Chunk boyutu (karakter)
            chunk_overlap: Chunk örtüşmesi (karakter)
            collection_name: ChromaDB koleksiyon adı
            persist_directory: Vektör DB kayıt dizini
            llm_model: Ollama model adı
            top_k: Her sorgu için döndürülecek chunk sayısı
        """
        print("\n" + "=" * 60)
        print("🚀 RAG PIPELINE BAŞLATILIYOR")
        print("=" * 60 + "\n")

        # Konfigürasyonu sakla
        self.config = {
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "collection_name": collection_name,
            "persist_directory": persist_directory,
            "llm_model": llm_model,
            "top_k": top_k,
        }

        # Bileşenleri başlat
        print("1️⃣  Document Loader hazırlanıyor...")
        self.loader = DocumentLoader()

        print("\n2️⃣  Text Chunker hazırlanıyor...")
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        print(f"\n3️⃣  Embedder hazırlanıyor ({embedding_model})...")
        self.embedder = Embedder(model_name=embedding_model)

        print(f"\n4️⃣  Vector Store hazırlanıyor ({collection_name})...")
        self.vector_store = VectorStore(
            collection_name=collection_name, persist_directory=persist_directory
        )

        print("\n5️⃣  Retriever hazırlanıyor...")
        self.retriever = Retriever(
            embedder=self.embedder, vector_store=self.vector_store, top_k=top_k
        )

        print(f"\n6️⃣  Generator hazırlanıyor ({llm_model})...")
        self.generator = Generator(model=llm_model)

        print("\n" + "=" * 60)
        print("✅ RAG PIPELINE HAZIR!")
        print("=" * 60 + "\n")

    def index_documents(self, source: str, clear_existing: bool = False) -> int:
        """
        Dökümanları indexler (embedding + vector store).

        Bu işlem bir kez yapılır. Dökümanlar vector store'a eklenir.

        Args:
            source: Döküman yolu (dosya veya dizin)
            clear_existing: Mevcut verileri sil

        Returns:
            İndexlenen chunk sayısı
        """
        print("\n" + "-" * 60)
        print("📥 INDEXING BAŞLIYOR")
        print("-" * 60)

        # Mevcut verileri temizle (opsiyonel)
        if clear_existing:
            print("\n🧹 Mevcut veriler temizleniyor...")
            self.vector_store.clear()

        # 1. Dökümanları yükle
        print("\n[Adım 1/4] Dökümanlar yükleniyor...")

        if os.path.isfile(source):
            docs = [self.loader.load_file(source)]
            docs = [d for d in docs if d]  # None'ları filtrele
        elif os.path.isdir(source):
            docs = self.loader.load_directory(source)
        else:
            # Doğrudan metin olarak kabul et
            docs = [self.loader.load_text(source, "direct_input")]

        if not docs:
            print("⚠️  Yüklenecek döküman bulunamadı!")
            return 0

        # 2. Chunk'la
        print("\n[Adım 2/4] Metinler parçalanıyor (chunking)...")
        chunks = self.chunker.chunk_documents(docs)

        if not chunks:
            print("⚠️  Oluşturulan chunk yok!")
            return 0

        # 3. Embed et
        print("\n[Adım 3/4] Embedding'ler oluşturuluyor...")
        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(contents)

        # 4. Vector store'a ekle
        print("\n[Adım 4/4] Vector store'a ekleniyor...")
        count = self.vector_store.add_documents(chunks, embeddings)

        print("\n" + "-" * 60)
        print(f"✅ INDEXING TAMAMLANDI: {count} chunk indexlendi")
        print("-" * 60 + "\n")

        return count

    def query(
        self, question: str, top_k: Optional[int] = None, return_sources: bool = True
    ) -> RAGResponse:
        """
        Soru sorar ve RAG ile yanıt alır.

        Args:
            question: Kullanıcı sorusu
            top_k: Kullanılacak chunk sayısı
            return_sources: Kaynak bilgisi ekle

        Returns:
            RAGResponse nesnesi
        """
        print("\n" + "-" * 60)
        print(f"❓ SORU: {question}")
        print("-" * 60)

        k = top_k or self.config["top_k"]

        # 1. Retrieval
        print("\n[Adım 1/2] İlgili bilgiler aranıyor...")
        retrieved = self.retriever.retrieve(question, top_k=k)

        if not retrieved:
            print("⚠️  İlgili bilgi bulunamadı!")
            return RAGResponse(
                answer="Üzgünüm, bu konuda bilgi bulamadım.",
                sources=[],
                retrieved_chunks=[],
            )

        # Context'i oluştur
        context = self.retriever.retrieve_with_context(question, top_k=k)

        # 2. Generation
        print("\n[Adım 2/2] Yanıt üretiliyor...")
        answer = self.generator.generate(question=question, context=context)

        # Kaynakları topla
        sources = []
        if return_sources:
            sources = self.retriever.get_sources(question, top_k=k)

        print("\n" + "-" * 60)
        print("✅ YANIT HAZIR")
        print("-" * 60 + "\n")

        return RAGResponse(answer=answer, sources=sources, retrieved_chunks=retrieved)

    def add_document(self, text: str, source_name: str = "manual_input") -> int:
        """
        Tek bir metni sisteme ekler.

        Args:
            text: Eklenecek metin
            source_name: Kaynak adı

        Returns:
            Eklenen chunk sayısı
        """
        doc = self.loader.load_text(text, source_name)
        chunks = self.chunker.chunk_documents([doc])

        contents = [chunk.content for chunk in chunks]
        embeddings = self.embedder.embed_texts(contents)

        return self.vector_store.add_documents(chunks, embeddings)

    def get_stats(self) -> Dict:
        """Sistem istatistiklerini döndürür."""
        return {
            "config": self.config,
            "vector_store": self.vector_store.get_stats(),
            "llm_available": self.generator.check_model_available(),
        }

    def clear(self):
        """Tüm indexlenmiş verileri siler."""
        self.vector_store.clear()
        print("🧹 Tüm veriler temizlendi")


# Hızlı başlangıç için yardımcı fonksiyon
def create_rag_pipeline(**kwargs) -> RAGPipeline:
    """
    RAG Pipeline oluşturmak için factory fonksiyonu.

    Örnek:
    >>> rag = create_rag_pipeline(llm_model="mistral", top_k=5)
    """
    return RAGPipeline(**kwargs)


# Test için
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RAG PIPELINE TEST")
    print("=" * 60)

    # Pipeline oluştur
    rag = RAGPipeline(
        persist_directory="./test_chroma_db", collection_name="test_collection"
    )

    # Test verisi ekle
    test_doc = """
    Python Programlama Dili
    
    Python, Guido van Rossum tarafından geliştirilen yüksek seviyeli 
    bir programlama dilidir. İlk sürümü 1991'de yayınlanmıştır.
    
    Python'un Özellikleri:
    - Okunabilir ve temiz sözdizimi
    - Dinamik tip sistemi
    - Geniş standart kütüphane
    - Çoklu paradigma desteği (OOP, fonksiyonel)
    
    Kullanım Alanları:
    - Web geliştirme (Django, Flask)
    - Veri bilimi (NumPy, Pandas)
    - Yapay zeka (TensorFlow, PyTorch)
    - Otomasyon ve scripting
    """

    # İndexle
    rag.index_documents(test_doc, clear_existing=True)

    # Soru sor
    response = rag.query("Python'u kim geliştirdi?")

    print("\n💬 YANIT:")
    print(response.answer)
    print(f"\n📚 Kaynaklar: {response.sources}")
