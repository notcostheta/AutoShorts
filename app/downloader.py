import yt_dlp
from pathlib import Path
import ffmpeg
import os


def download_video(url):
    """
    Download a video from a given URL and return the path to the downloaded file.

    Parameters:
        url (str): The URL of the video to be downloaded.

    Returns:
        Path: The path to the downloaded file.

    """
    temp_path = Path("content")
    temp_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    ydl_opts = {
        "outtmpl": str(temp_path / "%(title)s.%(ext)s"),
        "concurrent_fragment_downloads": 1,
        "max_concurrent_downloads": 1,
        # "format": "best[ext=mp4][vcodec=avc1]/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        # yt-dlp -F url
        # Take notes of the availible formats of the videos, I wonder how I couldn't find docs for this
        "format": "137+140/399+140/136+140/398+140/best",
        "merge_output_format": "mp4",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        output_file_path_str = ydl.prepare_filename(info_dict)
        # Convert the output file path from string to Path object
        output_file_path = Path(output_file_path_str)
        return output_file_path

    # Example usage
    # video_path = download_video("https://www.youtube.com/watch?v=XNQhDl4a9Ko")


def get_audio(video_path):
    """
    Extracts the audio from a video file and returns the path to the extracted audio file.

    Args:
        video_path (Path): The path to the video file.

    Returns:
        Path: The path to the extracted audio file.

    Raises:
        None

    """
    audio_path = video_path.with_suffix(".mp3")
    if audio_path.exists():
        return audio_path
    try:
        stream = ffmpeg.input(str(video_path))
        stream = ffmpeg.output(stream, str(audio_path))
        ffmpeg.run(stream)
    except Exception as e:
        raise Exception(f"Failed to extract audio: {e}")
    return audio_path

    # Example usage
    # audio_path = get_audio(video_path)
