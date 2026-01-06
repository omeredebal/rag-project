"""
Text Chunker - Metin Parçalama Modülü

Bu modül, uzun metinleri küçük, yönetilebilir parçalara (chunk) böler.

RAG Sistemindeki Rolü:
- Büyük dökümanları embedding için uygun boyuta getirir
- Overlap ile chunk'lar arası bağlamı korur
- Her chunk'a kaynak metadata'sını ekler

Neden Chunking Önemli?
1. Embedding modelleri genellikle max 512 token işler
2. Küçük parçalar daha spesifik bilgi içerir
3. Retrieval'da daha kesin sonuçlar verir
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Chunk:
    """Bir metin parçasını temsil eden veri sınıfı"""

    content: str  # Parça içeriği
    metadata: Dict = field(default_factory=dict)  # Ek bilgiler
    chunk_id: Optional[str] = None  # Benzersiz ID

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Chunk(id={self.chunk_id}, preview='{preview}')"


class TextChunker:
    """
    Metin Parçalayıcı

    İki strateji sunar:
    1. Karakter bazlı: Sabit karakter sayısına göre böler
    2. Cümle bazlı: Cümle sınırlarına göre böler

    Parametreler:
    - chunk_size: Her parçanın maksimum boyutu
    - chunk_overlap: Parçalar arası örtüşme miktarı

    Örnek:
    Metin: "A B C D E F G H I J" (chunk_size=4, overlap=2)
    Chunk 1: "A B C D"
    Chunk 2: "C D E F"  (C D örtüşüyor)
    Chunk 3: "E F G H"  (E F örtüşüyor)
    Chunk 4: "G H I J"  (G H örtüşüyor)
    """

    def __init__(
        self, chunk_size: int = 500, chunk_overlap: int = 50, separator: str = "\n\n"
    ):
        """
        Args:
            chunk_size: Her chunk'ın maksimum karakter sayısı
            chunk_overlap: Chunk'lar arası örtüşme (bağlam koruma)
            separator: Öncelikli bölme noktası
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Overlap, chunk_size'dan küçük olmalı!")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_text(self, text: str, metadata: Dict = None) -> List[Chunk]:
        """
        Metni chunk'lara böler.

        Args:
            text: Bölünecek metin
            metadata: Tüm chunk'lara eklenecek metadata

        Returns:
            Chunk listesi
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        chunks = []

        # Önce separator'a göre bölmeyi dene
        segments = text.split(self.separator)

        current_chunk = ""

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Segment tek başına chunk_size'ı aşıyorsa, karakter bazlı böl
            if len(segment) > self.chunk_size:
                # Önce mevcut chunk'ı kaydet
                if current_chunk:
                    chunks.append(
                        self._create_chunk(current_chunk, metadata, len(chunks))
                    )
                    current_chunk = ""

                # Büyük segmenti karakter bazlı böl
                char_chunks = self._split_by_characters(segment, metadata, len(chunks))
                chunks.extend(char_chunks)
                continue

            # Segment mevcut chunk'a sığıyor mu?
            test_chunk = (
                current_chunk + self.separator + segment if current_chunk else segment
            )

            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Mevcut chunk'ı kaydet
                if current_chunk:
                    chunks.append(
                        self._create_chunk(current_chunk, metadata, len(chunks))
                    )

                # Yeni chunk başlat (overlap ile)
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap :]
                    current_chunk = overlap_text + " " + segment
                else:
                    current_chunk = segment

        # Son chunk'ı kaydet
        if current_chunk.strip():
            chunks.append(self._create_chunk(current_chunk, metadata, len(chunks)))

        return chunks

    def _split_by_characters(
        self, text: str, metadata: Dict, start_index: int
    ) -> List[Chunk]:
        """
        Metni karakter sayısına göre böler (overlap ile).

        Bu metod uzun paragrafları işlemek için kullanılır.
        """
        chunks = []
        start = 0
        chunk_index = start_index

        while start < len(text):
            # Chunk'ın bitiş noktasını belirle
            end = start + self.chunk_size

            # Metinin sonuna geldik mi?
            if end >= len(text):
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                break

            # Kelime ortasında bölmemeye çalış
            # Son boşluğu bul
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk(chunk_text, metadata, chunk_index))
                chunk_index += 1

            # Sonraki chunk'ın başlangıcı (overlap ile)
            start = end - self.chunk_overlap

        return chunks

    def _create_chunk(self, content: str, base_metadata: Dict, index: int) -> Chunk:
        """Yeni bir Chunk nesnesi oluşturur."""
        chunk_metadata = {
            **base_metadata,
            "chunk_index": index,
            "chunk_size": len(content),
        }

        # Benzersiz ID oluştur
        source = base_metadata.get("source", "unknown")
        chunk_id = f"{source}_chunk_{index}"

        return Chunk(content=content, metadata=chunk_metadata, chunk_id=chunk_id)

    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Birden fazla dökümanı chunk'lara böler.

        Args:
            documents: Document nesnelerinin listesi

        Returns:
            Tüm chunk'ların listesi
        """
        all_chunks = []

        for doc_idx, doc in enumerate(documents):
            # Document nesnesinden içerik ve metadata al
            content = doc.content if hasattr(doc, "content") else str(doc)
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            # Döküman indexini metadata'ya ekle
            metadata["doc_index"] = doc_idx

            chunks = self.split_text(content, metadata)
            all_chunks.extend(chunks)

        print(f"📄 {len(documents)} dökümandan {len(all_chunks)} chunk oluşturuldu")
        return all_chunks


# Chunking stratejilerini görselleştiren yardımcı fonksiyon
def visualize_chunks(text: str, chunk_size: int = 100, overlap: int = 20):
    """Chunking'in nasıl çalıştığını görselleştirir."""

    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = chunker.split_text(text)

    print(f"\n{'='*60}")
    print(f"CHUNKING GÖRSELLEŞTİRME")
    print(f"{'='*60}")
    print(f"Orijinal metin uzunluğu: {len(text)} karakter")
    print(f"Chunk boyutu: {chunk_size}, Overlap: {overlap}")
    print(f"Oluşan chunk sayısı: {len(chunks)}")
    print(f"{'='*60}\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ({len(chunk.content)} karakter) ---")
        print(chunk.content)
        print()


# Test için
if __name__ == "__main__":
    sample_text = """
Python Programlama Dili

Python, yüksek seviyeli, genel amaçlı bir programlama dilidir. Guido van Rossum tarafından geliştirilmiş ve ilk sürümü 1991'de yayınlanmıştır.

Temel Özellikler

Python'un en önemli özellikleri arasında okunabilirlik, basit sözdizimi ve geniş kütüphane desteği yer alır. Dinamik tip sistemine sahiptir ve hem nesne yönelimli hem de fonksiyonel programlama paradigmalarını destekler.

Kullanım Alanları

Web geliştirme, veri bilimi, yapay zeka, otomasyon ve sistem yönetimi gibi pek çok alanda kullanılır. Django, Flask, NumPy, Pandas gibi popüler kütüphanelere sahiptir.
""".strip()

    visualize_chunks(sample_text, chunk_size=200, overlap=30)
