"""
RAG Sistemi - FastAPI Web Arayüzü
Modern, Responsive UI - HTML/Tailwind CSS/JS
"""

import os
import sys
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Sistemi")
rag_pipeline = None


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class IndexRequest(BaseModel):
    directory: str


class TextRequest(BaseModel):
    text: str
    source: str = "manual"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Sistemi</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        dark: { 800: '#1e1e1e', 900: '#121212', 700: '#2d2d2d' }
                    }
                }
            }
        }
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * { font-family: 'Inter', sans-serif; }
        .chat-area { height: calc(100vh - 180px); }
        @media (min-width: 1024px) { .chat-area { height: calc(100vh - 140px); } }
        .fade-in { animation: fadeIn 0.2s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
        .typing::after { content: ''; animation: blink 1s infinite; }
        @keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }
        .scrollbar-thin::-webkit-scrollbar { width: 4px; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: #d1d5db; border-radius: 2px; }
        .dark .scrollbar-thin::-webkit-scrollbar-thumb { background: #4b5563; }
    </style>
</head>
<body class="bg-gray-50 dark:bg-dark-900 min-h-screen transition-colors duration-200">
    
    <!-- Mobile Header -->
    <header class="lg:hidden bg-white dark:bg-dark-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 sticky top-0 z-50">
        <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
                <div class="w-8 h-8 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-lg flex items-center justify-center">
                    <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                    </svg>
                </div>
                <span class="font-semibold text-gray-900 dark:text-white">RAG</span>
            </div>
            <div class="flex items-center gap-2">
                <button onclick="toggleDarkMode()" class="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white">
                    <svg id="darkIcon" class="w-5 h-5 hidden dark:block" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/></svg>
                    <svg id="lightIcon" class="w-5 h-5 block dark:hidden" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/></svg>
                </button>
                <button onclick="toggleSidebar()" class="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7"/></svg>
                </button>
            </div>
        </div>
    </header>

    <div class="flex h-screen lg:h-screen">
        <!-- Sidebar -->
        <aside id="sidebar" class="fixed lg:static inset-y-0 left-0 z-40 w-72 bg-white dark:bg-dark-800 border-r border-gray-200 dark:border-gray-700 transform -translate-x-full lg:translate-x-0 transition-transform duration-200">
            <div class="flex flex-col h-full">
                <!-- Logo -->
                <div class="hidden lg:flex items-center gap-3 p-5 border-b border-gray-200 dark:border-gray-700">
                    <div class="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center">
                        <svg class="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="font-semibold text-gray-900 dark:text-white">RAG Sistemi</h1>
                        <p class="text-xs text-gray-500 dark:text-gray-400">Yerel AI Asistanı</p>
                    </div>
                </div>

                <div class="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin">
                    <!-- Status -->
                    <div class="flex items-center justify-between p-3 bg-gray-50 dark:bg-dark-700 rounded-xl">
                        <span class="text-sm text-gray-600 dark:text-gray-300">Durum</span>
                        <div class="flex items-center gap-1.5">
                            <div id="statusDot" class="w-2 h-2 bg-gray-400 rounded-full"></div>
                            <span id="statusText" class="text-sm font-medium text-gray-600 dark:text-gray-300">...</span>
                        </div>
                    </div>

                    <!-- Index -->
                    <div class="bg-gray-50 dark:bg-dark-700 rounded-xl p-4">
                        <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-3">Doküman İndeksle</h3>
                        <input type="text" id="indexPath" value="./data" 
                            class="w-full px-3 py-2 text-sm bg-white dark:bg-dark-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none mb-2">
                        <button onclick="indexDocuments()" id="indexBtn"
                            class="w-full bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium py-2 rounded-lg transition">
                            <span id="indexBtnText">İndeksle</span>
                        </button>
                        <p id="indexResult" class="text-xs mt-2 hidden"></p>
                    </div>

                    <!-- Stats -->
                    <div class="bg-gray-50 dark:bg-dark-700 rounded-xl p-4 space-y-2">
                        <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-2">Bilgi</h3>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-500 dark:text-gray-400">Model</span>
                            <span class="text-gray-900 dark:text-white font-medium">llama3.2</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-500 dark:text-gray-400">Embedding</span>
                            <span class="text-gray-900 dark:text-white font-medium">MiniLM-L6</span>
                        </div>
                        <div class="flex justify-between text-sm">
                            <span class="text-gray-500 dark:text-gray-400">Chunk</span>
                            <span id="docCount" class="text-gray-900 dark:text-white font-medium">0</span>
                        </div>
                    </div>

                    <!-- Settings -->
                    <div class="bg-gray-50 dark:bg-dark-700 rounded-xl p-4">
                        <h3 class="text-sm font-medium text-gray-900 dark:text-white mb-3">Ayarlar</h3>
                        <label class="text-xs text-gray-500 dark:text-gray-400">Top-K Sonuç</label>
                        <input type="number" id="topK" value="3" min="1" max="10"
                            class="w-full px-3 py-2 text-sm bg-white dark:bg-dark-800 border border-gray-200 dark:border-gray-600 rounded-lg text-gray-900 dark:text-white focus:ring-2 focus:ring-indigo-500 outline-none mt-1">
                    </div>
                </div>

                <!-- Footer -->
                <div class="p-4 border-t border-gray-200 dark:border-gray-700">
                    <div class="hidden lg:flex items-center justify-between">
                        <button onclick="toggleDarkMode()" class="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-white rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700">
                            <svg id="darkIconLg" class="w-5 h-5 hidden dark:block" fill="currentColor" viewBox="0 0 20 20"><path d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z"/></svg>
                            <svg id="lightIconLg" class="w-5 h-5 block dark:hidden" fill="currentColor" viewBox="0 0 20 20"><path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/></svg>
                        </button>
                        <button onclick="clearChat()" class="p-2 text-gray-500 hover:text-red-500 dark:text-gray-400 dark:hover:text-red-400 rounded-lg hover:bg-gray-100 dark:hover:bg-dark-700" title="Sohbeti Temizle">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"/></svg>
                        </button>
                    </div>
                </div>
            </div>
        </aside>

        <!-- Overlay -->
        <div id="overlay" onclick="toggleSidebar()" class="fixed inset-0 bg-black/50 z-30 hidden lg:hidden"></div>

        <!-- Main -->
        <main class="flex-1 flex flex-col min-w-0">
            <!-- Chat -->
            <div id="chatMessages" class="flex-1 overflow-y-auto p-4 lg:p-6 space-y-4 scrollbar-thin chat-area">
                <div id="welcome" class="flex items-center justify-center h-full">
                    <div class="text-center max-w-sm">
                        <div class="w-16 h-16 bg-gradient-to-br from-indigo-100 to-purple-100 dark:from-indigo-900/30 dark:to-purple-900/30 rounded-2xl flex items-center justify-center mx-auto mb-4">
                            <svg class="w-8 h-8 text-indigo-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"/>
                            </svg>
                        </div>
                        <h2 class="text-lg font-semibold text-gray-900 dark:text-white mb-2">Merhaba!</h2>
                        <p class="text-sm text-gray-500 dark:text-gray-400">Dokümanlarınızı indeksleyin ve sorularınızı sorun.</p>
                    </div>
                </div>
            </div>

            <!-- Input -->
            <div class="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-dark-800 p-4">
                <div class="max-w-4xl mx-auto flex gap-2">
                    <input type="text" id="queryInput" placeholder="Sorunuzu yazın..."
                        class="flex-1 px-4 py-3 bg-gray-50 dark:bg-dark-700 border border-gray-200 dark:border-gray-600 rounded-xl text-sm text-gray-900 dark:text-white placeholder-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none"
                        onkeypress="if(event.key==='Enter')sendQuery()">
                    <button onclick="sendQuery()" id="sendBtn"
                        class="px-5 py-3 bg-indigo-500 hover:bg-indigo-600 disabled:bg-indigo-300 text-white rounded-xl transition flex items-center gap-2">
                        <span class="hidden sm:inline text-sm font-medium">Gönder</span>
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"/>
                        </svg>
                    </button>
                </div>
            </div>
        </main>
    </div>

    <script>
        // Dark mode
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            document.documentElement.classList.add('dark');
        }

        function toggleDarkMode() {
            document.documentElement.classList.toggle('dark');
            localStorage.theme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('overlay');
            sidebar.classList.toggle('-translate-x-full');
            overlay.classList.toggle('hidden');
        }

        function clearChat() {
            const chat = document.getElementById('chatMessages');
            chat.innerHTML = document.getElementById('welcome').outerHTML;
        }

        // Status check
        async function checkStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();
                document.getElementById('statusDot').className = 'w-2 h-2 rounded-full ' + (data.ready ? 'bg-green-500' : 'bg-yellow-500');
                document.getElementById('statusText').textContent = data.ready ? 'Hazır' : 'Bekliyor';
                document.getElementById('docCount').textContent = data.doc_count;
            } catch {
                document.getElementById('statusDot').className = 'w-2 h-2 rounded-full bg-red-500';
                document.getElementById('statusText').textContent = 'Hata';
            }
        }
        checkStatus();

        // Index
        async function indexDocuments() {
            const path = document.getElementById('indexPath').value;
            const btn = document.getElementById('indexBtn');
            const btnText = document.getElementById('indexBtnText');
            const result = document.getElementById('indexResult');
            
            btn.disabled = true;
            btnText.textContent = 'İndeksleniyor...';
            
            try {
                const res = await fetch('/api/index', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ directory: path })
                });
                const data = await res.json();
                
                if (res.ok) {
                    result.className = 'text-xs mt-2 text-green-600 dark:text-green-400';
                    result.textContent = '✓ ' + data.chunks_indexed + ' chunk indekslendi';
                    checkStatus();
                } else {
                    result.className = 'text-xs mt-2 text-red-600 dark:text-red-400';
                    result.textContent = '✗ ' + (data.detail || 'Hata');
                }
                result.classList.remove('hidden');
            } catch {
                result.className = 'text-xs mt-2 text-red-600 dark:text-red-400';
                result.textContent = '✗ Bağlantı hatası';
                result.classList.remove('hidden');
            } finally {
                btn.disabled = false;
                btnText.textContent = 'İndeksle';
            }
        }

        // Query
        async function sendQuery() {
            const input = document.getElementById('queryInput');
            const query = input.value.trim();
            if (!query) return;

            const topK = parseInt(document.getElementById('topK').value) || 3;
            const chat = document.getElementById('chatMessages');
            const sendBtn = document.getElementById('sendBtn');
            const welcome = document.getElementById('welcome');
            
            if (welcome) welcome.remove();

            // User message
            const userDiv = document.createElement('div');
            userDiv.className = 'flex justify-end fade-in';
            userDiv.innerHTML = '<div class="max-w-[85%] lg:max-w-[70%] bg-indigo-500 text-white px-4 py-3 rounded-2xl rounded-br-sm"><p class="text-sm">' + escapeHtml(query) + '</p></div>';
            chat.appendChild(userDiv);

            // Loading
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'flex justify-start fade-in';
            loadingDiv.id = 'loading';
            loadingDiv.innerHTML = '<div class="bg-gray-100 dark:bg-dark-700 px-4 py-3 rounded-2xl rounded-bl-sm"><p class="text-sm text-gray-500 dark:text-gray-400">Düşünüyor<span class="typing">...</span></p></div>';
            chat.appendChild(loadingDiv);

            input.value = '';
            sendBtn.disabled = true;
            chat.scrollTop = chat.scrollHeight;

            const startTime = Date.now();

            try {
                const res = await fetch('/api/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query, top_k: topK })
                });
                const data = await res.json();
                const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

                document.getElementById('loading')?.remove();

                const assistantDiv = document.createElement('div');
                assistantDiv.className = 'flex justify-start fade-in';

                if (res.ok) {
                    let contextHtml = '';
                    if (data.contexts?.length) {
                        contextHtml = '<details class="mt-3 text-xs"><summary class="text-gray-400 dark:text-gray-500 cursor-pointer hover:text-gray-600 dark:hover:text-gray-300">📚 ' + data.contexts.length + ' kaynak · ' + elapsed + 's</summary><div class="mt-2 space-y-1">';
                        data.contexts.forEach((ctx, i) => {
                            contextHtml += '<div class="p-2 bg-gray-50 dark:bg-dark-800 rounded text-gray-500 dark:text-gray-400 truncate">' + escapeHtml(ctx.source) + '</div>';
                        });
                        contextHtml += '</div></details>';
                    }

                    assistantDiv.innerHTML = `
                        <div class="max-w-[85%] lg:max-w-[70%] bg-gray-100 dark:bg-dark-700 px-4 py-3 rounded-2xl rounded-bl-sm group relative">
                            <p class="text-sm text-gray-800 dark:text-gray-200 whitespace-pre-wrap">${escapeHtml(data.answer)}</p>
                            ${contextHtml}
                            <button onclick="copyText(this)" class="absolute top-2 right-2 p-1.5 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 opacity-0 group-hover:opacity-100 transition" title="Kopyala">
                                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>
                            </button>
                        </div>`;
                } else {
                    assistantDiv.innerHTML = '<div class="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-4 py-3 rounded-2xl rounded-bl-sm text-sm">⚠️ ' + (data.detail || 'Hata') + '</div>';
                }
                chat.appendChild(assistantDiv);
            } catch {
                document.getElementById('loading')?.remove();
                const errorDiv = document.createElement('div');
                errorDiv.className = 'flex justify-start fade-in';
                errorDiv.innerHTML = '<div class="bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 px-4 py-3 rounded-2xl rounded-bl-sm text-sm">⚠️ Bağlantı hatası</div>';
                chat.appendChild(errorDiv);
            } finally {
                sendBtn.disabled = false;
                chat.scrollTop = chat.scrollHeight;
            }
        }

        function copyText(btn) {
            const text = btn.parentElement.querySelector('p').textContent;
            navigator.clipboard.writeText(text);
            btn.innerHTML = '<svg class="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/></svg>';
            setTimeout(() => {
                btn.innerHTML = '<svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/></svg>';
            }, 1500);
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

    if not os.path.exists(request.directory):
        raise HTTPException(
            status_code=400, detail=f"Klasör bulunamadı: {request.directory}"
        )

    try:
        num_chunks = rag_pipeline.index_documents(request.directory)
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
        raise HTTPException(status_code=400, detail="Önce doküman indeksleyin")

    try:
        result = rag_pipeline.query(
            request.query, top_k=request.top_k, return_sources=True
        )
        contexts = [
            {"text": c.content, "source": c.metadata.get("source", "?")}
            for c in result.retrieved_chunks
        ]
        return {"answer": result.answer, "contexts": contexts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_index():
    global rag_pipeline
    if rag_pipeline:
        rag_pipeline.clear()
    return {"success": True}


def main():
    global rag_pipeline
    print("\n" + "=" * 50)
    print("🚀 RAG Sistemi Başlatılıyor")
    print("=" * 50 + "\n")

    rag_pipeline = RAGPipeline(collection_name="rag_collection", llm_model="llama3.2")

    print("\n" + "=" * 50)
    print("✅ http://localhost:8000")
    print("=" * 50 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
