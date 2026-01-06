#!/usr/bin/env python3
"""
RAG Demo Script

Bu script, RAG sisteminin nasıl çalıştığını gösterir.
Adım adım tüm pipeline'ı çalıştırır.

Kullanım:
    python demo.py
"""

import os
import sys

# Proje kök dizinini path'e ekle
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_banner():
    """Hoşgeldin mesajı"""
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🔍 RAG (Retrieval-Augmented Generation) Demo             ║
║                                                              ║
║     Bu demo, RAG sisteminin tüm bileşenlerini                ║
║     adım adım gösterir.                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    )


def step1_document_loading():
    """Adım 1: Döküman Yükleme"""
    print("\n" + "=" * 60)
    print("📚 ADIM 1: DÖKÜMAN YÜKLEME (Document Loading)")
    print("=" * 60)

    print(
        """
    Document Loader, ham dökümanları sisteme alır.
    - Dosyaları okur (.txt, .md)
    - Metadata ekler (dosya adı, boyut, tarih)
    - Sonraki aşamaya hazırlar
    """
    )

    from src.document_loader import DocumentLoader

    loader = DocumentLoader()
    docs = loader.load_directory("data/")

    print(f"\n📊 Sonuç: {len(docs)} döküman yüklendi")

    for doc in docs:
        print(
            f"   - {doc.metadata.get('filename')}: {doc.metadata.get('char_count')} karakter"
        )

    return docs


def step2_chunking(docs):
    """Adım 2: Metin Parçalama"""
    print("\n" + "=" * 60)
    print("✂️  ADIM 2: METİN PARÇALAMA (Chunking)")
    print("=" * 60)

    print(
        """
    Chunking, uzun metinleri küçük parçalara böler.
    - Embedding modelleri için uygun boyut (max ~500 karakter)
    - Overlap ile bağlam korunur
    - Her chunk bağımsız aranabilir olur
    """
    )

    from src.chunker import TextChunker

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_documents(docs)

    print(f"\n📊 Sonuç: {len(chunks)} chunk oluşturuldu")
    print("\n📝 İlk 3 chunk:")

    for i, chunk in enumerate(chunks[:3], 1):
        preview = chunk.content[:100].replace("\n", " ")
        print(f"\n   Chunk {i} ({len(chunk.content)} karakter):")
        print(f"   '{preview}...'")

    return chunks


def step3_embedding(chunks):
    """Adım 3: Embedding Oluşturma"""
    print("\n" + "=" * 60)
    print("🔢 ADIM 3: EMBEDDING OLUŞTURMA (Vectorization)")
    print("=" * 60)

    print(
        """
    Embedding, metni sayısal vektörlere dönüştürür.
    - Her chunk → 384 boyutlu vektör
    - Benzer anlamlar → Yakın vektörler
    - Semantic arama mümkün olur
    """
    )

    from src.embedder import Embedder

    embedder = Embedder()

    # İlk chunk'ı embed et ve göster
    sample_embedding = embedder.embed_text(chunks[0].content)

    print(f"\n📊 Embedding boyutu: {len(sample_embedding)}")
    print(f"   İlk 10 değer: {[round(x, 4) for x in sample_embedding[:10]]}")

    # Tüm chunk'ları embed et
    contents = [c.content for c in chunks]
    all_embeddings = embedder.embed_texts(contents)

    print(f"\n✅ {len(all_embeddings)} chunk embed edildi")

    return embedder, all_embeddings


def step4_vector_store(chunks, embeddings):
    """Adım 4: Vektör Depolama"""
    print("\n" + "=" * 60)
    print("💾 ADIM 4: VEKTÖR DEPOLAMA (Vector Store)")
    print("=" * 60)

    print(
        """
    Vector Store, embedding'leri depolar ve aranabilir kılar.
    - ChromaDB kullanıyoruz (lokal, ücretsiz)
    - Kalıcı depolama
    - Hızlı benzerlik araması
    """
    )

    from src.vector_store import VectorStore

    store = VectorStore(
        collection_name="demo_collection", persist_directory="./demo_chroma_db"
    )

    # Temizle ve ekle
    store.clear()
    count = store.add_documents(chunks, embeddings)

    stats = store.get_stats()
    print(f"\n📊 Depolanan chunk sayısı: {stats['document_count']}")

    return store


def step5_retrieval(embedder, store):
    """Adım 5: Bilgi Getirme"""
    print("\n" + "=" * 60)
    print("🔍 ADIM 5: BİLGİ GETİRME (Retrieval)")
    print("=" * 60)

    print(
        """
    Retrieval, sorguya en uygun chunk'ları getirir.
    - Sorgu embed edilir
    - Vector store'da benzerlik araması
    - En alakalı K chunk döndürülür
    """
    )

    from src.retriever import Retriever

    retriever = Retriever(embedder=embedder, vector_store=store, top_k=3)

    # Test sorgusu
    test_query = "Python programlama dili nedir ve ne için kullanılır?"
    print(f"\n❓ Test Sorgusu: '{test_query}'")

    results = retriever.retrieve(test_query)

    print(f"\n📊 Bulunan {len(results)} sonuç:")

    for i, r in enumerate(results, 1):
        print(f"\n   [{i}] Skor: {r.score:.3f}")
        preview = r.content[:150].replace("\n", " ")
        print(f"       '{preview}...'")

    return retriever


def step6_generation(retriever):
    """Adım 6: Yanıt Üretme"""
    print("\n" + "=" * 60)
    print("🤖 ADIM 6: YANIT ÜRETME (Generation)")
    print("=" * 60)

    print(
        """
    Generation, LLM ile yanıt üretir.
    - Retrieved context LLM'e verilir
    - LLM, context'e dayalı yanıt üretir
    - Ollama ile lokal LLM kullanıyoruz
    """
    )

    from src.generator import Generator

    generator = Generator(model="llama3.2")

    # Test sorusu ve context
    question = "Python'un temel özellikleri nelerdir?"
    context = retriever.retrieve_with_context(question)

    print(f"\n❓ Soru: '{question}'")
    print(f"\n📄 Context (kısaltılmış):\n   {context[:300]}...")

    print("\n⏳ LLM yanıt üretiyor...")
    answer = generator.generate(question=question, context=context)

    print(f"\n💬 YANIT:\n{answer}")

    return generator


def step7_full_pipeline():
    """Adım 7: Tam Pipeline Demo"""
    print("\n" + "=" * 60)
    print("🚀 ADIM 7: TAM RAG PIPELINE")
    print("=" * 60)

    print(
        """
    Şimdi tüm bileşenleri birleştiren RAGPipeline sınıfını
    kullanarak end-to-end demo yapacağız.
    """
    )

    from src.rag_pipeline import RAGPipeline

    # Pipeline oluştur
    rag = RAGPipeline(
        collection_name="full_demo", persist_directory="./full_demo_db", top_k=3
    )

    # Dökümanları indexle
    rag.index_documents("data/", clear_existing=True)

    # Sorular sor
    questions = [
        "Python'u kim geliştirdi?",
        "Makine öğrenmesi türleri nelerdir?",
        "RAG nedir ve nasıl çalışır?",
        "Derin öğrenme için hangi kütüphaneler kullanılır?",
    ]

    print("\n" + "-" * 60)
    print("📋 SORU-CEVAP DEMOsu")
    print("-" * 60)

    for q in questions:
        response = rag.query(q)

        print(f"\n❓ SORU: {q}")
        print(f"\n💬 YANIT: {response.answer}")
        print(f"\n📚 Kaynaklar: {response.sources}")
        print("\n" + "." * 60)

    return rag


def interactive_mode(rag):
    """İnteraktif soru-cevap modu"""
    print("\n" + "=" * 60)
    print("💬 İNTERAKTİF MOD")
    print("=" * 60)
    print(
        """
    Artık kendi sorularınızı sorabilirsiniz!
    Çıkmak için 'q' veya 'exit' yazın.
    """
    )

    while True:
        try:
            question = input("\n❓ Sorunuz: ").strip()

            if question.lower() in ["q", "exit", "quit", "çık"]:
                print("\n👋 Görüşürüz!")
                break

            if not question:
                continue

            response = rag.query(question)
            print(f"\n💬 YANIT:\n{response.answer}")

            if response.sources:
                print(f"\n📚 Kaynaklar: {', '.join(response.sources)}")

        except KeyboardInterrupt:
            print("\n\n👋 Görüşürüz!")
            break


def main():
    """Ana demo fonksiyonu"""
    print_banner()

    try:
        # Adım adım demo
        docs = step1_document_loading()
        chunks = step2_chunking(docs)
        embedder, embeddings = step3_embedding(chunks)
        store = step4_vector_store(chunks, embeddings)
        retriever = step5_retrieval(embedder, store)
        generator = step6_generation(retriever)
        rag = step7_full_pipeline()

        # İnteraktif mod
        print("\n" + "=" * 60)
        print("✅ DEMO TAMAMLANDI!")
        print("=" * 60)

        try_interactive = (
            input("\n🎮 İnteraktif moda geçmek ister misiniz? (e/h): ").strip().lower()
        )

        if try_interactive in ["e", "evet", "y", "yes"]:
            interactive_mode(rag)

    except ImportError as e:
        print(f"\n❌ Eksik paket hatası: {e}")
        print("\nLütfen gereken paketleri yükleyin:")
        print("   pip install -r requirements.txt")
        print("\nOllama için:")
        print("   brew install ollama")
        print("   ollama pull llama3.2")

    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
