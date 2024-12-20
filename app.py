import faiss
import numpy as np
import streamlit as st
import os
import yt_dlp
import subprocess
import whisper
import json
import re  # Regular expressions for sanitizing filenames
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import warnings  # Suppress PyTorch warnings

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Function to sanitize file names to avoid issues with special characters
def sanitize_filename(filename):
    """Sanitizes the filename by replacing special characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

# Function to download a YouTube video and extract its audio
def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """Downloads a YouTube video, extracts its audio, and returns the audio file path and sanitized title."""
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save with title as filename
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_title = info.get('title', 'video')
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None, None

    sanitized_video_title = sanitize_filename(video_title)
    video_file = os.path.join(video_folder, f"{sanitized_video_title}.mp4")
    audio_file = os.path.join(audio_folder, f"{sanitized_video_title}.mp3")

    try:
        extract_audio(video_file, audio_file)
        return audio_file, sanitized_video_title
    except Exception as e:
        st.error(f"Failed to extract audio: {e}")
        return None, None
    finally:
        if os.path.exists(video_file):
            os.remove(video_file)

# Function to extract audio from the video using FFmpeg
def extract_audio(video_path, audio_path):
    """Extracts audio from a video file using FFmpeg."""
    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    try:
        subprocess.run(command, check=True)
    except Exception as e:
        st.error(f"FFmpeg failed to extract audio: {e}")

# Function to transcribe the audio using Whisper
def transcribe_audio(audio_path, model_name="base"):
    """Transcribes audio using OpenAI's Whisper."""
    try:
        model = whisper.load_model(model_name, download_root='./whisper-models/')
        result = model.transcribe(audio_path)
        return result['text']
    except Exception as e:
        st.error(f"Failed to transcribe audio: {e}")
        return None
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Initialize FAISS database
def initialize_faiss_database(embedding_dim):
    """Initializes a FAISS database for storing embeddings."""
    return faiss.IndexFlatL2(embedding_dim)

# Generate embeddings using SentenceTransformer
def generate_embeddings(text, model_name="all-MiniLM-L6-v2"):
    """Generates embeddings for the given text using SentenceTransformers."""
    model = SentenceTransformer(model_name)
    embeddings = np.array(model.encode([text]), dtype='float32')
    return embeddings

# Add embeddings to FAISS index
def add_to_faiss(index, embeddings, metadata, metadata_file="metadata.json"):
    """Adds embeddings and metadata to the FAISS database and saves metadata to a file."""
    index.add(embeddings)
    with open(metadata_file, "w") as file:
        json.dump(metadata, file)

# Search FAISS for the most relevant results
def search_faiss(index, query_embedding, top_k=3):
    """Searches the FAISS database for the closest embeddings to the query."""
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]

# Load metadata for the videos
def load_metadata(metadata_file="metadata.json"):
    """Loads metadata from a JSON file."""
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as file:
            return json.load(file)
    else:
        return []

# Retrieve the most relevant context using FAISS
def retrieve_context(index, query, metadata_file="metadata.json", model_name="all-MiniLM-L6-v2"):
    """Retrieves the most relevant context for a given user query."""
    query_embedding = generate_embeddings(query, model_name)
    indices, _ = search_faiss(index, query_embedding)
    metadata = load_metadata(metadata_file)
    results = [metadata[idx]['transcription'] for idx in indices if idx < len(metadata)]
    return " ".join(results)

# Use HuggingFace Transformers to generate answers
def generate_answer_with_huggingface(context, query):
    """Generates an answer using HuggingFace Transformers."""
    model_name = "google/flan-t5-base"
    question_answering_pipeline = pipeline("text2text-generation", model=model_name)
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = question_answering_pipeline(prompt, max_length=100, num_return_sequences=1)
    return response[0]["generated_text"]

# Streamlit User Interface
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

# User input for YouTube video URL
youtube_url = st.text_input("Enter YouTube Video URL")

if youtube_url:
    with st.spinner("Downloading and processing video..."):
        audio_file, video_title = download_video(youtube_url)
    
    if audio_file:
        with st.spinner("Transcribing audio..."):
            transcription = transcribe_audio(audio_file)

        if transcription:
            with st.spinner("Initializing FAISS database..."):
                faiss_index = initialize_faiss_database(384)
                embedding = generate_embeddings(transcription)
                add_to_faiss(faiss_index, embedding, [{"video_title": video_title, "transcription": transcription}])

            # User input for a question about the video
            query = st.text_input("Ask a question about the video:")

            if query:
                with st.spinner("Retrieving context and generating answer..."):
                    context = retrieve_context(faiss_index, query)
                    answer = generate_answer_with_huggingface(context, query)
                    st.write(f"**Answer:** {answer}")
