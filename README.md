
# 🎬 YouTube Q&A Chatbot (TubeTalk Bot)

## 📌 Project Goal

The **TubeTalk Bot** is an interactive AI chatbot that allows users to input a YouTube video URL and ask questions about the video through **text or voice**. It transcribes the video audio, builds a searchable vector database, and responds with accurate answers or summaries using **LangChain Agents** and **OpenAI GPT-4**.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
  A[User Input (YouTube URL)] --> B[Download Audio using yt_dlp]
  B --> C[Transcribe using Whisper]
  C --> D[Split Text into Chunks]
  D --> E[Build Chroma Vectorstore]
  E --> F[LangChain Agent (GPT-4)]
  F --> G[Answer Question or Summarize]
  G --> H[Display in Chainlit Interface]
```

---

## 🧪 Methodology

- **Audio Extraction**: Extract high-quality audio from YouTube using `yt_dlp`.
- **Transcription**: Use OpenAI's `whisper` model for multilingual, accurate audio transcription.
- **Chunking & Embedding**: Split transcript into chunks using HuggingFace tokenizer and encode with `MiniLM`.
- **Vector Store**: Store vectorized chunks in a local ChromaDB for semantic search.
- **LangChain Agent**: Use a custom agent with tools for summarization and question answering.
- **Chainlit UI**: Real-time chat interface with support for text and voice input/output.

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-ai-chatbot.git
cd youtube-ai-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file with the following:

```
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
```

---

## 📁 Repository Structure

```
.
├── newapp.py               # Main Chainlit application
├── requirements.txt        # Dependencies
├── sessions/               # Temporary session storage (auto-created)
├── saved_audio/            # Stores recorded voice input
└── .env                    # API keys (not committed)
```

---

## 🚀 Running the App

```bash
chainlit run newapp.py --port 8501
```

Visit `http://localhost:8501` in your browser to use the chatbot.

---

## 🗣️ Usage Guide

1. **Enter a YouTube link** — e.g., `https://www.youtube.com/watch?v=abcd1234`.
2. Wait while the video is processed (audio extracted, transcribed, and embedded).
3. Ask any question related to the video content via text or **record your voice**.
4. The bot will:
   - Use a summarization tool if you're asking for a summary.
   - Use a search tool for context-based answers from the transcript.

---

## 📎 Example Prompts

- "Summarize the video for me."
- "What did the speaker say about climate change?"
- "ماذا قال عن الذكاء الاصطناعي؟"

---

## ✅ Features

- 🎧 Voice-to-text input using `Whisper`
- 🔍 Semantic search with `Chroma`
- 🤖 GPT-4 powered responses with LangChain Agent tools
- 🌐 Multilingual support
- 🧠 Summarization capability
- ⚡ Real-time responses via Chainlit
