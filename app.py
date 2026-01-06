"""
RAG Sistemi - FastAPI Web Arayüzü
Profesyonel ve Sade UI - HTML/Tailwind CSS/JS
"""

import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Sistemi")

# Global RAG Pipeline
rag_pipeline = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class IndexRequest(BaseModel):
    directory: str


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Sistemi</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif; }
        .chat-container { height: calc(100vh - 280px); }
        .message-enter { animation: fadeIn 0.3s ease-out; }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .loading-dots::after {
            content: '';
            animation: dots 1.5s steps(4, end) infinite;
        }
        @keyframes dots {
            0%, 20% { content: ''; }
            40% { content: '.'; }
            60% { content: '..'; }
            80%, 100% { content: '...'; }
        }
        .context-box { max-height: 150px; overflow-y: auto; }
        pre { white-space: pre-wrap; word-wrap: break-word; }
    </style>
</head>
<body class="bg-gray-50 min-h-screen">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div class="max-w-6xl mx-auto px-6 py-4">
            <div class="flex items-center justify-between">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-xl font-semibold text-gray-900">RAG Sistemi</h1>
                        <p class="text-sm text-gray-500">Yerel AI Asistanı</p>
                    </div>
                </div>
                <div id="status" class="flex items-center gap-2 px-3 py-1.5 bg-gray-100 rounded-full">
                    <div id="status-dot" class="w-2 h-2 bg-gray-400 rounded-full"></div>
                    <span id="status-text" class="text-sm text-gray-600">Bağlanıyor...</span>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-6 py-6">
        <div class="grid grid-cols-1 lg:grid-cols-4 gap-6">
            <!-- Sidebar -->
            <aside class="lg:col-span-1 space-y-4">
                <!-- Index Card -->
                <div class="bg-white rounded-2xl border border-gray-200 p-5">
                    <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                        <svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"/>
                        </svg>
                        Doküman İndeksleme
                    </h3>
                    <div class="space-y-3">
                        <div>
                            <label class="block text-sm font-medium text-gray-700 mb-1.5">Klasör Yolu</label>
                            <input type="text" id="indexPath" value="./data" 
                                   class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                        </div>
                        <button onclick="indexDocuments()" id="indexBtn"
                                class="w-full bg-blue-500 hover:bg-blue-600 text-white font-medium py-2 px-4 rounded-lg transition flex items-center justify-center gap-2">
                            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
                            </svg>
                            <span id="indexBtnText">İndeksle</span>
                        </button>
                    </div>
                    <div id="indexResult" class="mt-3 text-sm hidden"></div>
                </div>

                <!-- Stats Card -->
                <div class="bg-white rounded-2xl border border-gray-200 p-5">
                    <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                        <svg class="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/>
                        </svg>
                        Sistem Bilgisi
                    </h3>
                    <div class="space-y-3 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-500">Model</span>
                            <span class="font-medium text-gray-900" id="modelName">llama3.2</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Embedding</span>
                            <span class="font-medium text-gray-900">MiniLM-L6-v2</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">İndekslenen</span>
                            <span class="font-medium text-gray-900" id="docCount">0 chunk</span>
                        </div>
                    </div>
                </div>

                <!-- Settings Card -->
                <div class="bg-white rounded-2xl border border-gray-200 p-5">
                    <h3 class="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                        <svg class="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/>
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/>
                        </svg>
                        Ayarlar
                    </h3>
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1.5">Top-K Sonuç</label>
                        <input type="number" id="topK" value="3" min="1" max="10"
                               class="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition">
                        <p class="text-xs text-gray-400 mt-1">Getirilecek bağlam sayısı</p>
                    </div>
                </div>
            </aside>

            <!-- Chat Area -->
            <div class="lg:col-span-3">
                <div class="bg-white rounded-2xl border border-gray-200 h-full flex flex-col">
                    <!-- Chat Messages -->
                    <div id="chatMessages" class="chat-container overflow-y-auto p-6 space-y-4">
                        <div class="text-center py-12">
                            <div class="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                                <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                                </svg>
                            </div>
                            <h3 class="text-lg font-medium text-gray-900 mb-2">RAG Sistemi Hazır</h3>
                            <p class="text-gray-500 text-sm max-w-md mx-auto">
                                Dokümanlarınızı indeksleyin ve sorularınızı sorun. 
                                Sistem, ilgili bağlamı bulup yanıt oluşturacak.
                            </p>
                        </div>
                    </div>

                    <!-- Input Area -->
                    <div class="border-t border-gray-200 p-4">
                        <div class="flex gap-3">
                            <input type="text" id="queryInput" placeholder="Sorunuzu yazın..."
                                   class="flex-1 px-4 py-3 border border-gray-300 rounded-xl text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition"
                                   onkeypress="if(event.key === 'Enter') sendQuery()">
                            <button onclick="sendQuery()" id="sendBtn"
                                    class="bg-blue-500 hover:bg-blue-600 text-white font-medium py-3 px-6 rounded-xl transition flex items-center gap-2">
                                <span>Gönder</span>
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                                </svg>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <script>
        // Check status on load
        document.addEventListener('DOMContentLoaded', () => {
            checkStatus();
        });

        async function checkStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');
                const docCount = document.getElementById('docCount');
                
                if (data.ready) {
                    statusDot.className = 'w-2 h-2 bg-green-500 rounded-full';
                    statusText.textContent = 'Hazır';
                    statusText.className = 'text-sm text-green-600';
                    docCount.textContent = data.doc_count + ' chunk';
                } else {
                    statusDot.className = 'w-2 h-2 bg-yellow-500 rounded-full';
                    statusText.textContent = 'Başlatılıyor...';
                    statusText.className = 'text-sm text-yellow-600';
                }
            } catch (error) {
                const statusDot = document.getElementById('status-dot');
                const statusText = document.getElementById('status-text');
                statusDot.className = 'w-2 h-2 bg-red-500 rounded-full';
                statusText.textContent = 'Bağlantı Hatası';
                statusText.className = 'text-sm text-red-600';
            }
        }

        async function indexDocuments() {
            const path = document.getElementById('indexPath').value;
            const btn = document.getElementById('indexBtn');
            const btnText = document.getElementById('indexBtnText');
            const result = document.getElementById('indexResult');
            
            btn.disabled = true;
            btnText.innerHTML = '<span class="loading-dots">İndeksleniyor</span>';
            
            try {
                const response = await fetch('/api/index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ directory: path })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    result.className = 'mt-3 text-sm p-3 bg-green-50 text-green-700 rounded-lg';
                    result.innerHTML = '✓ ' + data.chunks_indexed + ' chunk indekslendi';
                    document.getElementById('docCount').textContent = data.total_chunks + ' chunk';
                    checkStatus();
                } else {
                    result.className = 'mt-3 text-sm p-3 bg-red-50 text-red-700 rounded-lg';
                    result.innerHTML = '✗ ' + (data.detail || 'Hata oluştu');
                }
                result.classList.remove('hidden');
            } catch (error) {
                result.className = 'mt-3 text-sm p-3 bg-red-50 text-red-700 rounded-lg';
                result.innerHTML = '✗ Bağlantı hatası';
                result.classList.remove('hidden');
            } finally {
                btn.disabled = false;
                btnText.textContent = 'İndeksle';
            }
        }

        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;
            
            const topK = parseInt(document.getElementById('topK').value) || 3;
            const messagesDiv = document.getElementById('chatMessages');
            const sendBtn = document.getElementById('sendBtn');
            
            // Clear welcome message if exists
            if (messagesDiv.querySelector('.text-center')) {
                messagesDiv.innerHTML = '';
            }
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'flex justify-end message-enter';
            userMsg.innerHTML = '<div class="max-w-[80%] bg-blue-500 text-white px-4 py-3 rounded-2xl rounded-br-md"><p class="text-sm">' + escapeHtml(query) + '</p></div>';
            messagesDiv.appendChild(userMsg);
            
            // Add loading message
            const loadingMsg = document.createElement('div');
            loadingMsg.className = 'flex justify-start message-enter';
            loadingMsg.id = 'loading-msg';
            loadingMsg.innerHTML = '<div class="max-w-[80%] bg-gray-100 px-4 py-3 rounded-2xl rounded-bl-md"><p class="text-sm text-gray-600 loading-dots">Düşünüyor</p></div>';
            messagesDiv.appendChild(loadingMsg);
            
            input.value = '';
            sendBtn.disabled = true;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, top_k: topK })
                });
                
                const data = await response.json();
                
                // Remove loading message
                const loadingEl = document.getElementById('loading-msg');
                if (loadingEl) loadingEl.remove();
                
                if (response.ok) {
                    // Add assistant message
                    const assistantMsg = document.createElement('div');
                    assistantMsg.className = 'flex justify-start message-enter';
                    
                    let contextHtml = '';
                    if (data.contexts && data.contexts.length > 0) {
                        contextHtml = '<details class="mt-3"><summary class="text-xs text-gray-500 cursor-pointer hover:text-gray-700">📚 ' + data.contexts.length + ' bağlam kullanıldı</summary><div class="mt-2 space-y-2">';
                        data.contexts.forEach(function(ctx, i) {
                            const source = ctx.source || 'Kaynak ' + (i+1);
                            const text = ctx.text.length > 300 ? ctx.text.substring(0, 300) + '...' : ctx.text;
                            contextHtml += '<div class="context-box text-xs bg-gray-50 p-2 rounded border border-gray-200"><div class="font-medium text-gray-600 mb-1">' + escapeHtml(source) + '</div><pre class="text-gray-500">' + escapeHtml(text) + '</pre></div>';
                        });
                        contextHtml += '</div></details>';
                    }
                    
                    assistantMsg.innerHTML = '<div class="max-w-[80%] bg-gray-100 px-4 py-3 rounded-2xl rounded-bl-md"><p class="text-sm text-gray-800 whitespace-pre-wrap">' + escapeHtml(data.answer) + '</p>' + contextHtml + '</div>';
                    messagesDiv.appendChild(assistantMsg);
                } else {
                    const errorMsg = document.createElement('div');
                    errorMsg.className = 'flex justify-start message-enter';
                    errorMsg.innerHTML = '<div class="max-w-[80%] bg-red-50 text-red-700 px-4 py-3 rounded-2xl rounded-bl-md"><p class="text-sm">⚠️ ' + (data.detail || 'Bir hata oluştu') + '</p></div>';
                    messagesDiv.appendChild(errorMsg);
                }
            } catch (error) {
                const loadingEl = document.getElementById('loading-msg');
                if (loadingEl) loadingEl.remove();
                const errorMsg = document.createElement('div');
                errorMsg.className = 'flex justify-start message-enter';
                errorMsg.innerHTML = '<div class="max-w-[80%] bg-red-50 text-red-700 px-4 py-3 rounded-2xl rounded-bl-md"><p class="text-sm">⚠️ Bağlantı hatası</p></div>';
                messagesDiv.appendChild(errorMsg);
            } finally {
                sendBtn.disabled = false;
                messagesDiv.scrollTop = messagesDiv.scrollHeight;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@app.get("/api/status")
async def get_status():
    global rag_pipeline
    if rag_pipeline is None:
        return {"ready": False, "doc_count": 0}

    doc_count = rag_pipeline.vector_store.collection.count()
    return {"ready": True, "doc_count": doc_count}


@app.post("/api/index")
async def index_documents(request: IndexRequest):
    global rag_pipeline

    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline başlatılmadı")

    directory = request.directory
    if not os.path.exists(directory):
        raise HTTPException(status_code=400, detail=f"Klasör bulunamadı: {directory}")

    try:
        num_chunks = rag_pipeline.index_documents(directory)
        total_chunks = rag_pipeline.vector_store.collection.count()
        return {
            "success": True,
            "chunks_indexed": num_chunks,
            "total_chunks": total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query")
async def query(request: QueryRequest):
    global rag_pipeline

    if rag_pipeline is None:
        raise HTTPException(status_code=500, detail="RAG Pipeline başlatılmadı")

    if rag_pipeline.vector_store.collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Henüz doküman indekslenmedi. Önce dokümanları indeksleyin.",
        )

    try:
        result = rag_pipeline.query(
            request.query, top_k=request.top_k, return_sources=True
        )

        contexts = []
        for chunk in result.retrieved_chunks:
            contexts.append(
                {
                    "text": chunk.content,
                    "source": (
                        chunk.metadata.get("source", "Bilinmeyen")
                        if chunk.metadata
                        else "Bilinmeyen"
                    ),
                }
            )

        return {"answer": result.answer, "contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global rag_pipeline

    print("\n" + "=" * 60)
    print("🚀 RAG Web Arayüzü Başlatılıyor")
    print("=" * 60 + "\n")

    # Initialize RAG Pipeline
    rag_pipeline = RAGPipeline(
        collection_name="rag_web_collection", llm_model="llama3.2"
    )

    print("\n" + "=" * 60)
    print("✅ Sunucu hazır: http://localhost:8000")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
