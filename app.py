import os
import re
import yt_dlp
import subprocess
import streamlit as st

def sanitize_filename(filename):
    """Sanitizes the filename by replacing special characters with underscores."""
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def get_downloaded_file_path(directory, title):
    """Returns the actual file path of the downloaded file with any extension."""
    sanitized_title = sanitize_filename(title)
    for ext in ['mp4', 'm4a', 'webm']:
        file_path = os.path.join(directory, f"{sanitized_title}.{ext}")
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError(f"Video file not found for title: {sanitized_title}")

def download_video(youtube_url, video_folder="video_downloads", audio_folder="audio_downloads"):
    """Downloads a YouTube video and extracts audio. Renames the file to avoid special characters."""
    video_folder = os.path.abspath(video_folder)
    audio_folder = os.path.abspath(audio_folder)
    os.makedirs(video_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(video_folder, '%(title)s.%(ext)s'),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_title = info.get('title', 'video')

    video_file = get_downloaded_file_path(video_folder, video_title)
    audio_file = os.path.join(audio_folder, f"{sanitize_filename(video_title)}.mp3")

    return audio_file, video_file, sanitize_filename(video_title)

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
        raise RuntimeError(f"Audio extraction failed: {e.stderr.decode()}")

# Streamlit App
st.title("RAG-based Video Q&A System")
st.subheader("Enter YouTube Video URL and ask questions")

youtube_url = st.text_input("Enter YouTube Video URL")

if youtube_url:
    st.write("Downloading video and extracting audio...")
    try:
        audio_file, video_file, video_title = download_video(youtube_url)
        st.write(f"Video file found: {video_file}")
        extract_audio(video_file, audio_file)
        st.write(f"Audio extracted: {audio_file}")
    except Exception as e:
        st.error(f"Error occurred during video/audio processing: {e}")
        st.stop()

