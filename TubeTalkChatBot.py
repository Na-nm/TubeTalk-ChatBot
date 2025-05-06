# chainlit run newapp.py --port 8501

import os
import shutil
import uuid
import tempfile
import asyncio
from io import BytesIO
import whisper
import yt_dlp
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
import chainlit as cl
from chainlit.element import Element
import numpy as np
import soundfile as sf  # NEW: for saving audio files in proper format

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

# Load models
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
whisper_model = whisper.load_model("base")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "TubeTalk bot"

# Global vars
session_dir = None
vectorstore = None
full_transcript = ""

# Session management
def create_new_session():
    global session_dir
    session_dir = os.path.join("sessions", str(uuid.uuid4()))
    os.makedirs(session_dir, exist_ok=True)

def clean_session():
    global session_dir, vectorstore, full_transcript
    if vectorstore:
        try:
            vectorstore._collection = None
            vectorstore = None
        except:
            pass
    if session_dir and os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    session_dir = None
    full_transcript = ""

# YouTube audio downloading
def download_audio_from_youtube(url):
    output_template = os.path.join(session_dir, 'audio')
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)
    return output_template + ".mp3"

# Transcription
def transcribe_audio(filepath):
    result = whisper_model.transcribe(filepath)
    return result["text"], result["language"]

# Text splitting
def tokenize_and_chunk_hf(text, max_tokens=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return [tokenizer.decode(tokens[i:i + max_tokens]).strip() for i in range(0, len(tokens), max_tokens)]

# Vectorstore building
def build_vectorstore(chunks):
    persist_directory = os.path.join(session_dir, "chroma_db")
    db = Chroma.from_texts(chunks, embedding_model, persist_directory=persist_directory)
    db.persist()
    return db

# Summarization
def summarize_text(transcript):
    try:
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0.3)
        prompt = f"""
        Summarize the following video transcript in a concise and informative way.

        Transcript:
        {transcript}

        Summary:
        """
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"⚠️ Summarization failed: {str(e)}"

# Agent tools
def build_agent(vectorstore, transcript):
    def summarize_wrapper(_):
        transcript = cl.user_session.get("full_transcript")
        return summarize_text(transcript)

    def search_wrapper(question):
        if vectorstore is None:
            return "❌ Vectorstore is not available. Please upload a YouTube video first."
        results = vectorstore.similarity_search(question, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    tools = [
        Tool( # summarization tool
        name="Summarizer",
        func=summarize_wrapper,
        description="Use this tool to summarize the transcript that extracted from the video in short sentences."
        ),

        Tool( # search tool
        name="Search",
        func=search_wrapper,
        description="Use this tool to search for answers from the video transcript,\
            if the question is not related to the video transcript response with:\
            This question isn't related to the video, ask another question."
            )
    ]

    return initialize_agent(
        tools, 
        ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4"),
        agent="zero-shot-react-description", 
        verbose=True
        )

# Chainlit chat start
@cl.on_chat_start
async def start():
    clean_session()             
    create_new_session()  
    await cl.Message(content="👋 Hello! Please enter the YouTube URL to process the video.").send()
    cl.user_session.set("state", "awaiting_link")
    cl.user_session.set("agent", None)

# Handle user messages
@cl.on_message
async def handle_message(message: cl.Message):
    state = cl.user_session.get("state")
    agent = cl.user_session.get("agent")

    if state == "awaiting_link":
        try:
            # Check if the message is a valid YouTube link
            if not message.content.startswith("https://www.youtube.com/watch?v="):
                await cl.Message(content="🚫 Please enter a valid YouTube link.").send()
                return

            url = message.content
            #clean_session()
            #create_new_session()
            await cl.Message(content="🔄 Downloading audio from video...").send()
            audio_file = download_audio_from_youtube(url) 

            #await cl.Message(content="🔄 Transcribing audio...").send()
            transcript, _ = await asyncio.to_thread(transcribe_audio, audio_file)
            cl.user_session.set("full_transcript", transcript)

            #await cl.Message(content="🔄 Splitting text into chunks...").send()
            chunks = await asyncio.to_thread(tokenize_and_chunk_hf, transcript)

            #await cl.Message(content="🔄 Building vectorstore...").send()
            db = await asyncio.to_thread(build_vectorstore, chunks)
            cl.user_session.set("vectorstore", db)

            agent = build_agent(db, transcript)
            cl.user_session.set("agent", agent)
            cl.user_session.set("state", "ready_for_questions")

            await cl.Message(content="✅ Video processed! Ask your question by text or voice.").send()

        except Exception as e:
            await cl.Message(content=f"❌ Error processing video: {str(e)}").send()

    elif state == "ready_for_questions":
        try:
            await cl.Message(content="🤔 Thinking...").send()
            response = agent.invoke(message.content)

            if isinstance(response, dict) and "output" in response:
                await cl.Message(content=response["output"]).send()
            elif hasattr(response, "content"):
                await cl.Message(content=response.content).send()
            else:
                await cl.Message(content=str(response)).send()

        except Exception as e:
            await cl.Message(content=f"⚠️ Error answering: {str(e)}").send()
    else:
        await cl.Message(content="🚫 Please enter a YouTube link first.").send()

# -------------------------------
# 🎤 Voice Input Handling Section
# -------------------------------
@cl.on_audio_start
async def on_audio_start():
    await cl.Message(content="🎤 Start recording...").send()
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):

    if chunk.isStart:
        buffer = BytesIO()
        extension = "wav"
        if chunk.mimeType and "/" in chunk.mimeType:
            parts = chunk.mimeType.split("/")
            if len(parts) > 1:
                extension = parts[1]
        buffer.name = f"input_audio.{extension}"
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    buffer = cl.user_session.get("audio_buffer")
    if buffer:
        buffer.write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements: list[Element] = None):
    elements = elements or []

    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    if not audio_buffer:
        await cl.Message(content="⚠️ Doesn't recording anything, Please try again ").send()
        return

    audio_buffer.seek(0)
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type") or "audio/wav"

    whisper_input = (audio_buffer.name, audio_file, audio_mime_type)
    transcription = await speech_to_text(whisper_input)

    await cl.Message(
        author="You",
        type="user_message",
        content=transcription
    ).send()

    cl.user_session.set("state", "ready_for_questions")

    msg = cl.Message(author="You", content=transcription, elements=[])
    await handle_message(message=msg)


async def speech_to_text(whisper_input):
    file_name, file_bytes, mime_type = whisper_input

    os.makedirs("saved_audio", exist_ok=True)
    temp_path = f"saved_audio/{file_name}"

    audio_data = np.frombuffer(file_bytes, dtype=np.int16)
    sf.write(temp_path, audio_data, samplerate=16000, format='WAV', subtype='PCM_16')

    vc_transcript = whisper_model.transcribe(temp_path)["text"]

    return vc_transcript