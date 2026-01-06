"""
Generator - LLM ile Yanıt Üretme Modülü

Bu modül, retrieval sonuçlarını kullanarak LLM ile yanıt üretir.

RAG Sistemindeki Rolü:
- Retrieved context'i LLM'e verir
- Kullanıcı sorusunu yanıtlar
- Prompt engineering yapar

Kullanılan Teknoloji: Ollama
- Lokal LLM çalıştırma
- Ücretsiz ve gizlilik dostu
- Çeşitli model desteği (llama, mistral vb.)

Prompt Template Önemli!
- Sisteme rol/talimat verir
- Context'i doğru formatta sunar
- Yanıt kalitesini etkiler
"""

from typing import Optional, Dict, Any


class Generator:
    """
    LLM Yanıt Üretici

    Ollama API'si ile lokal LLM kullanarak
    context-based yanıtlar üretir.

    Örnek kullanım:
    >>> generator = Generator(model="llama3.2")
    >>> response = generator.generate(
    ...     question="Python nedir?",
    ...     context="Python yüksek seviyeli bir programlama dilidir..."
    ... )
    >>> print(response)
    """

    # Varsayılan prompt template (context ile)
    DEFAULT_TEMPLATE = """Sen yardımcı bir asistansın. Sana verilen bağlam bilgisini kullanarak soruyu yanıtla.

KURALLAR:
1. SADECE verilen bağlam bilgisini kullan
2. Bağlamda olmayan bilgiyi uydurma
3. Emin değilsen "Bu konuda bilgim yok" de
4. Yanıtı Türkçe ver
5. Kısa ve öz ol

BAĞLAM:
{context}

SORU: {question}

YANIT:"""

    # Context olmadan sohbet template
    CHAT_TEMPLATE = """Sen yardımcı bir Türkçe asistansın. Kullanıcının mesajına kısa ve samimi yanıt ver.

SORU: {question}

YANIT:"""

    def __init__(
        self,
        model: str = "llama3.2",
        template: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
    ):
        """
        Args:
            model: Ollama model adı (llama3.2, mistral, vb.)
            template: Özel prompt template (opsiyonel)
            temperature: Yaratıcılık seviyesi (0-1)
            max_tokens: Maksimum yanıt uzunluğu
        """
        self.model = model
        self.template = template or self.DEFAULT_TEMPLATE
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        self._initialize_client()

    def _initialize_client(self):
        """Ollama client'ını başlatır."""
        try:
            import ollama

            self.client = ollama
            print(f"✅ Ollama bağlantısı hazır (model: {self.model})")
        except ImportError:
            print("⚠️  ollama paketi bulunamadı!")
            print("   Kurulum: pip install ollama")
            print("   Ayrıca Ollama uygulamasının çalıştığından emin olun.")
            self.client = None

    def generate(self, question: str, context: str, stream: bool = False) -> str:
        """
        Soru ve context'e göre yanıt üretir.

        Args:
            question: Kullanıcı sorusu
            context: Retrieved bilgi (chunk'lar)
            stream: Streaming modu (opsiyonel)

        Returns:
            LLM yanıtı
        """
        if not self.client:
            return self._fallback_response(question, context)

        # Context varsa RAG template, yoksa sohbet template kullan
        if context and context.strip():
            prompt = self.template.format(context=context, question=question)
        else:
            prompt = self.CHAT_TEMPLATE.format(question=question)

        print(f"🤖 LLM yanıt üretiyor ({self.model})...")

        try:
            if stream:
                return self._generate_stream(prompt)
            else:
                return self._generate_sync(prompt)

        except Exception as e:
            print(f"❌ LLM hatası: {e}")
            return self._fallback_response(question, context)

    def _generate_sync(self, prompt: str) -> str:
        """Senkron yanıt üretimi."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        )

        return response["response"].strip()

    def _generate_stream(self, prompt: str) -> str:
        """Streaming yanıt üretimi."""
        full_response = ""

        for chunk in self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={"temperature": self.temperature, "num_predict": self.max_tokens},
        ):
            text = chunk["response"]
            print(text, end="", flush=True)
            full_response += text

        print()  # Yeni satır
        return full_response.strip()

    def _fallback_response(self, question: str, context: str) -> str:
        """
        Ollama çalışmadığında basit fallback yanıt.

        Bu, sistemin çalışmasını test etmek için kullanılır.
        Gerçek projede LLM aktif olmalı.
        """
        print("⚠️  Fallback mod: LLM olmadan basit yanıt")

        if not context:
            return "Üzgünüm, bu konuda bilgi bulamadım."

        # Basit extractive yanıt: İlk context parçasını döndür
        first_chunk = context.split("---")[0].strip()
        if "[Kaynak" in first_chunk:
            # Kaynak etiketini kaldır
            lines = first_chunk.split("\n")
            first_chunk = "\n".join(lines[1:]).strip()

        return f"Bulduğum bilgiye göre:\n\n{first_chunk[:500]}..."

    def check_model_available(self) -> bool:
        """Model'in Ollama'da yüklü olup olmadığını kontrol eder."""
        if not self.client:
            return False

        try:
            models = self.client.list()
            model_names = [m["name"] for m in models.get("models", [])]

            # Model adı veya model:tag formatını kontrol et
            for name in model_names:
                if self.model in name or name.startswith(self.model):
                    return True

            return False

        except Exception:
            return False

    def set_template(self, template: str):
        """Prompt template'i değiştirir."""
        self.template = template
        print("✅ Prompt template güncellendi")


# Farklı kullanım senaryoları için template'ler
TEMPLATES = {
    "default": Generator.DEFAULT_TEMPLATE,
    "concise": """Verilen bağlama göre soruyu kısaca yanıtla.

Bağlam: {context}

Soru: {question}

Kısa Yanıt:""",
    "detailed": """Sen bir uzman asistansın. Aşağıdaki bağlam bilgisini kullanarak 
soruyu detaylı ve açıklayıcı bir şekilde yanıtla.

=== BAĞLAM ===
{context}

=== SORU ===
{question}

=== DETAYLI YANIT ===
""",
    "qa_with_sources": """Soruyu yanıtla ve kaynaklarını belirt.

Bağlam:
{context}

Soru: {question}

Yanıt (kaynaklarla birlikte):""",
}


# Test için
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GENERATOR TEST")
    print("=" * 60)

    generator = Generator(model="llama3.2")

    # Model kontrolü
    if generator.check_model_available():
        print(f"✅ Model mevcut: {generator.model}")

        # Test yanıtı
        test_context = """
        Python, Guido van Rossum tarafından geliştirilen yüksek seviyeli 
        bir programlama dilidir. 1991'de ilk sürümü yayınlanmıştır.
        Okunabilirliği ve basit sözdizimi ile bilinir.
        """

        response = generator.generate(
            question="Python'u kim geliştirdi?", context=test_context
        )

        print(f"\n💬 Yanıt:\n{response}")
    else:
        print(f"⚠️  Model bulunamadı: {generator.model}")
        print("   Kurulum: ollama pull llama3.2")
