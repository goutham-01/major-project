import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
import os
import yt_dlp
import subprocess
import whisper
import streamlit as st
import re  # Regular expressions for sanitizing filenames


def sanitize_filename(filename):
    """Sanitize the filename by replacing special characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)


def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """Download a YouTube video and extract its audio."""
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',  # Download only the audio
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save as title.ext
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get('title', 'video')

    sanitized_video_title = sanitize_filename(video_title)
    video_file = os.path.join(video_folder, f"{sanitized_video_title}.mp4")
    audio_file = os.path.join(audio_folder, f"{sanitized_video_title}.mp3")

    extract_audio(video_file, audio_file)
    return audio_file, sanitized_video_title


def extract_audio(video_path, audio_path):
    """Extract audio from the video using FFmpeg."""
    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)

    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")


def transcribe_audio(audio_path, model_name="base"):
    """Transcribe audio using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result['text']


# Streamlit App
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

youtube_url = st.text_input("Enter YouTube Video URL")

if youtube_url:
    st.write("Downloading video and extracting audio...")
    try:
        audio_file, video_title = download_video(youtube_url)
        st.write(f"Audio extracted: {audio_file}")
    except Exception as e:
        st.error(f"Error occurred during video/audio processing: {e}")
        st.stop()

    st.write("Transcribing audio...")
    try:
        transcription = transcribe_audio(audio_file)
        st.write(f"Transcription: {transcription[:200]}...")  # Print first 200 characters of transcription
    except Exception as e:
        st.error(f"Error during transcription: {e}")
