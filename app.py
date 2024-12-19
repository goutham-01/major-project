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
import re

def sanitize_filename(filename):
    """Sanitizes the filename by replacing special characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """Downloads a YouTube video and extracts audio. Renames the file to avoid special characters."""
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',  # Download only the audio
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),  # Save with title as filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get('title', 'video')

    sanitized_video_title = sanitize_filename(video_title)
    video_file = os.path.join(video_folder, f"{sanitized_video_title}.mp4")
    audio_file = os.path.join(audio_folder, f"{sanitized_video_title}.mp3")

    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found at {video_file}")

    return audio_file, video_file, sanitized_video_title


def extract_audio(video_path, audio_path):
    """Extracts audio from a video file using FFmpeg."""
    video_path = os.path.abspath(video_path)
    audio_path = os.path.abspath(audio_path)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    command = ["ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")


# Streamlit App
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

youtube_url = st.text_input("Enter YouTube Video URL")

if youtube_url:
    st.write("Downloading video and extracting audio...")
    try:
        audio_file, video_file, video_title = download_video(youtube_url)
        extract_audio(video_file, audio_file)
        st.write(f"Audio extracted: {audio_file}")
    except Exception as e:
        st.error(f"Error occurred during video/audio processing: {e}")
        st.stop()
