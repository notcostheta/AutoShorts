import whisperx
from pathlib import Path


def transcribe_audio(
    audio_path,
    hf_read_key,
    device="cpu",
    batch_size=4,
    compute_type="int8",
    model_dir=".cache/whisperx/",
    model_name="large-v2",
):
    """
    Transcribes audio using WhisperX models, aligns the output, and assigns speaker labels.
    Example usage
    audio_path = Path("content/video.mp3")
    result = transcribe_audio(audio_path)
    print(result)

    Parameters:
    - audio_path: Path to the audio file to be transcribed.
    - device: Device to run the model on ("cpu" or "cuda"), default "cpu".
    - batch_size: Batch size for processing (higher for GPUs), default 4.
    - compute_type: Type of computation ("int8" for CPUs, "float16" for GPUs).
    - model_dir: Directory to store downloaded models.
    - model_name: Name of the WhisperX model to use (default "large-v2").

    Returns:
    - aligned_transcript: The final transcription result with speaker labels.
    """
    # Ensure the model directory exists using pathlib
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Load the WhisperX model
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type, download_root=model_dir
    )
    # Load the audio file
    audio = whisperx.load_audio(audio_path)
    # 1. Transcribe with original whisper (batched)
    result = model.transcribe(audio, batch_size=batch_size)

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"], device=device
    )
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # 3. Assign speaker labels
    hf_read = hf_read_key
    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_read, device=device)
    diarize_segments = diarize_model(audio)
    aligned_transcript = whisperx.assign_word_speakers(diarize_segments, result)

    return aligned_transcript

    # Example usage
    # audio_path = Path("content/video.mp3")
    # result = transcribe_audio(audio_path)
    # print(result)
