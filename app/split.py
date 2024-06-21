from pathlib import Path
import pandas as pd
import subprocess


def generate_ffmpeg_command(input_path, output_path, start_time, end_time):
    """
    Generate an FFmpeg command to split a video file.

    Args:
        input_path (str): The path to the input video file.
        output_path (str): The path to the output video file.
        start_time (str): The start time of the segment to be extracted in the format HH:MM:SS.
        end_time (str): The end time of the segment to be extracted in the format HH:MM:SS.

    Returns:
        str: The FFmpeg command to split the video.

    Example:
        >>> generate_ffmpeg_command('input.mp4', 'output.mp4', '00:00:10', '00:01:30')
        "ffmpeg -ss 00:00:10 -to 00:01:30 -i 'input.mp4' -c copy -y 'output.mp4'"
    """
    return f'ffmpeg -ss {start_time} -to {end_time} -i "{input_path}" -c copy -y "{output_path}"'


def process_video_chunks(data, video_path, output_dir):
    """
    Process video chunks based on the given data.

    Args:
        data (pandas.DataFrame): The data containing start and end timestamps for each chunk.
        video_path (str): The path to the input video file.
        output_dir (str): The directory where the output chunks will be saved.

    Returns:
        list: A list of output paths for the processed video chunks.
    """
    output_paths = []  # Initialize an empty list for output paths
    start_list = data["start"].tolist()
    end_list = data["end"].tolist()
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, (start, end) in enumerate(zip(start_list, end_list)):
        output_path = Path(output_dir) / f"{i}.mp4"
        command = generate_ffmpeg_command(video_path, output_path, start, end)
        print(command)
        subprocess.run(command, shell=True)
        print(f"Chunk {i} done")
        output_paths.append(str(output_path))  # Append the output path as a string
    return output_paths  # Return the list of output paths


def make_splits(data, video_path, output_dir):
    """
    Split the video into chunks based on the start and end timestamps in the given data.

    Args:
        data (pandas.DataFrame): The dataframe containing the start and end timestamps.
        video_path (str): The path to the video file.
        output_dir (str): The directory where the output chunks will be saved.

    Returns:
        list: A list of output paths for the video chunks.

    Raises:
        None

    """
    if "start" in data.columns and "end" in data.columns:
        output_paths = process_video_chunks(
            data, video_path, output_dir
        )  # Capture and return the output paths
        return output_paths
    else:
        print("Dataframe does not have the required columns.")
        return []


# Example usage
# video_path = Path("content/Google's secret algorithm exposed via leak to GitHub….mp4")
# data = pd.read_json("fireship_readable_chunks.json")
# output_dir = "output"
# output_paths = make_splits(data, video_path, output_dir)
# print(output_paths)
