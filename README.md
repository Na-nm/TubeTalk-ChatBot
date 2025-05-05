
# 🎬 YouTube Q&A Chatbot (TubeTalk Bot)

## 📌 Project Goal

The **TubeTalk Bot** is an interactive AI chatbot that allows users to input a YouTube video URL and ask questions about the video through **text or voice**. It transcribes the video audio, builds a searchable vector database, and responds with accurate answers or summaries using **LangChain Agents** and **OpenAI GPT-4**.

---

## 🏗️ Architecture Overview

1. **User Input**: The user submits a YouTube link.
2. **Audio Extraction**: Audio is downloaded from the video using `yt_dlp`.
3. **Transcription**: Audio is transcribed to text using the `Whisper` model.
4. **Chunking**: The transcript is split into manageable text chunks.
5. **Vector Storage**: The chunks are embedded and stored in a Chroma vector database.
6. **Agent Reasoning**: A LangChain agent (powered by GPT-4) uses tools to either:
   - Summarize the transcript.
   - Search for relevant content to answer user questions.
7. **User Interaction**: The response is delivered through the Chainlit interface.

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
