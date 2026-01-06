"""
Document Loader - Döküman Yükleme Modülü

Bu modül, farklı formatlardaki dökümanları okur ve
metadata ile birlikte döndürür.

RAG Sistemindeki Rolü:
- Ham veriyi sisteme alır
- Her döküman için metadata oluşturur
- Sonraki aşama (Chunking) için hazırlar
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Bir dökümanı temsil eden veri sınıfı"""

    content: str  # Dökümanın içeriği
    metadata: Dict = field(default_factory=dict)  # Ek bilgiler

    def __repr__(self):
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Document(preview='{preview}', metadata={self.metadata})"


class DocumentLoader:
    """
    Döküman Yükleyici

    Desteklenen formatlar:
    - .txt (düz metin)
    - .md (markdown)

    Örnek kullanım:
    >>> loader = DocumentLoader()
    >>> docs = loader.load_directory("data/")
    >>> print(f"{len(docs)} döküman yüklendi")
    """

    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    def __init__(self, encoding: str = "utf-8"):
        """
        Args:
            encoding: Dosya kodlaması (varsayılan: utf-8)
        """
        self.encoding = encoding

    def load_file(self, file_path: str) -> Optional[Document]:
        """
        Tek bir dosyayı yükler.

        Args:
            file_path: Dosya yolu

        Returns:
            Document nesnesi veya None (hata durumunda)
        """
        path = Path(file_path)

        # Dosya var mı kontrol et
        if not path.exists():
            print(f"⚠️  Dosya bulunamadı: {file_path}")
            return None

        # Uzantı destekleniyor mu?
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            print(f"⚠️  Desteklenmeyen format: {path.suffix}")
            return None

        try:
            # Dosyayı oku
            content = path.read_text(encoding=self.encoding)

            # Metadata oluştur
            metadata = {
                "source": str(path.absolute()),
                "filename": path.name,
                "extension": path.suffix,
                "size_bytes": path.stat().st_size,
                "loaded_at": datetime.now().isoformat(),
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
            }

            print(f"✅ Yüklendi: {path.name} ({len(content)} karakter)")
            return Document(content=content, metadata=metadata)

        except Exception as e:
            print(f"❌ Okuma hatası ({file_path}): {e}")
            return None

    def load_directory(
        self, directory_path: str, recursive: bool = True
    ) -> List[Document]:
        """
        Bir dizindeki tüm desteklenen dosyaları yükler.

        Args:
            directory_path: Dizin yolu
            recursive: Alt dizinlere de bak (varsayılan: True)

        Returns:
            Document listesi
        """
        path = Path(directory_path)

        if not path.exists():
            print(f"❌ Dizin bulunamadı: {directory_path}")
            return []

        if not path.is_dir():
            print(f"❌ Bu bir dizin değil: {directory_path}")
            return []

        documents = []

        # Dosyaları bul
        pattern = "**/*" if recursive else "*"
        files = [
            f
            for f in path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]

        print(f"\n📂 {len(files)} dosya bulundu: {directory_path}")
        print("-" * 40)

        for file_path in sorted(files):
            doc = self.load_file(str(file_path))
            if doc:
                documents.append(doc)

        print("-" * 40)
        print(f"📊 Toplam: {len(documents)} döküman yüklendi\n")

        return documents

    def load_text(self, text: str, source: str = "direct_input") -> Document:
        """
        Doğrudan metin girişinden döküman oluşturur.

        Args:
            text: Metin içeriği
            source: Kaynak adı

        Returns:
            Document nesnesi
        """
        metadata = {
            "source": source,
            "filename": source,
            "extension": None,
            "loaded_at": datetime.now().isoformat(),
            "char_count": len(text),
            "line_count": text.count("\n") + 1,
        }

        return Document(content=text, metadata=metadata)


# Test için
if __name__ == "__main__":
    loader = DocumentLoader()

    # Örnek metin
    sample_text = """
    Python, genel amaçlı bir programlama dilidir.
    Guido van Rossum tarafından geliştirilmiştir.
    Okunabilirliği ve basit sözdizimi ile bilinir.
    """

    doc = loader.load_text(sample_text.strip(), "test_document")
    print(f"Oluşturulan döküman: {doc}")
    print(f"Metadata: {doc.metadata}")
