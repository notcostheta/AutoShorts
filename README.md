# AutoShorts
Automated shortform content from Youtube Videos
Url in Shorts Out

## Requirements
- Make sure you have ffmpeg installed on your system. If not, you can install it by running `sudo apt install ffmpeg` on Ubuntu.
- If on windows, make sure to add ffmpeg to your PATH.
- Create a virtual environment by running `python -m venv venv`.
- Load the virtual environment by running `source venv/bin/activate`.
- Install the required python packages by running `pip install -r requirements.txt`.
- Create `"config.env"` file in the root directory and add the following:
```
HF_READ_TOKEN = "YOUR_HF_READ_TOKEN"
```
- Get your Hugging Face API token from [here](https://huggingface.co/login) and replace `YOUR_HF_READ_TOKEN` with your token.
- Run `run.ipyb` and replace the `video_url` with the URL of the video you want to generate shortform content for.