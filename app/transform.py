import pandas as pd

from semantic_chunkers import StatisticalChunker
from semantic_router.encoders import HuggingFaceEncoder


def split_on_max_time(word_list, max_duration):
    """
    Splits a word list into datasets based on a maximum duration.

    Args:
        word_list (list): A list of words with corresponding start and end times.
        max_duration (float, optional): The maximum duration of each dataset in seconds. Defaults to 15.

    Returns:
        list: A list of datasets, where each dataset is a DataFrame containing a subset of the word list.

    """
    wldf = pd.DataFrame(word_list)
    split_chars = ["and", ".", ","]
    datasets = []

    while (wldf.iloc[-1]["end"] - wldf.iloc[0]["start"]) > max_duration:
        mid_segment_length = (wldf.iloc[-1]["end"] - wldf.iloc[0]["start"]) / 2

        closest_row = None
        closest_diff = float("inf")

        for i in range(len(wldf)):
            if any(char in wldf.iloc[i]["word"] for char in split_chars):
                segment_length = wldf.iloc[i]["end"] - wldf.iloc[0]["start"]
                diff = abs(segment_length - mid_segment_length)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_row = i

        if closest_row is not None:
            datasets.append(wldf.iloc[: closest_row + 1])
            wldf = wldf.iloc[closest_row + 1 :]
        else:
            break

    datasets.append(wldf)
    return datasets


def process_datasets_internal(split_datasets):
    """
    Process the split datasets and return a list of dictionaries containing information about each dataset.

    Args:
        split_datasets (list): A list of split datasets.

    Returns:
        list: A list of dictionaries containing information about each dataset. Each dictionary contains the following keys:
            - "start": The start time of the dataset.
            - "end": The end time of the dataset.
            - "text": The concatenated text of all words in the dataset.
            - "words": A list of dictionaries representing each word in the dataset.
            - "speaker": The speaker of the dataset.
            - "segment_length": The length of the dataset in seconds.

    """
    result = []

    for split_dataset_iter in split_datasets:
        if split_dataset_iter.empty:
            continue  # Skip empty datasets to avoid errors

        sample_json = {
            "start": split_dataset_iter.iloc[0]["start"],
            "end": split_dataset_iter.iloc[-1]["end"],
            "text": " ".join(split_dataset_iter["word"]),
            "words": split_dataset_iter.to_dict(orient="records"),
            "speaker": split_dataset_iter.iloc[0]["speaker"],
            "segment_length": split_dataset_iter.iloc[-1]["end"]
            - split_dataset_iter.iloc[0]["start"],
        }
        result.append(sample_json)

    return result


def process_datasets_external(split_datasets):
    """
    Process the split datasets and generate a list of sample JSON objects.

    Args:
        split_datasets (list): A list of split datasets.

    Returns:
        list: A list of sample JSON objects.

    """
    result = []

    for split_dataset_iter in split_datasets:
        if split_dataset_iter.empty:
            continue  # Skip empty datasets to avoid errors

        # Directly perform operations within the loop
        sample_json = {
            "text": " ".join(split_dataset_iter["word"]),
            "start": split_dataset_iter.iloc[0]["start"],
            "end": split_dataset_iter.iloc[-1]["end"],
            "words": split_dataset_iter.to_dict(orient="records"),
            "chunk_length": split_dataset_iter.iloc[-1]["end"]
            - split_dataset_iter.iloc[0]["start"],
            "speaker": split_dataset_iter.iloc[0]["speaker"],
        }
        result.append(sample_json)

    return result


def process_extra_indices_internal(df, extra_indices, max_duration):
    """
    Process extra indices in the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        extra_indices (list): A list of indices to process.

    Returns:
        pandas.DataFrame: The processed DataFrame.

    """
    new_df = df.copy()
    for index in extra_indices:
        word_list = df.loc[index]["words"]
        split_datasets = split_on_max_time(word_list, max_duration)
        processed_datasets = process_datasets_internal(split_datasets)
        new_df = new_df.drop(index)
        new_df = pd.concat([new_df, pd.DataFrame(processed_datasets)])
    return new_df


def process_extra_indices_external(df, extra_indices, max_duration):
    """
    Process extra indices in the given DataFrame by splitting the word list, processing the datasets,
    and updating the DataFrame accordingly.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        extra_indices (list): A list of indices to process.

    Returns:
        pandas.DataFrame: The updated DataFrame after processing the extra indices.
    """
    new_df = df.copy()
    for index in extra_indices:
        word_list = df.loc[index]["words"]
        split_datasets = split_on_max_time(word_list, max_duration)
        processed_datasets = process_datasets_external(split_datasets)
        new_df = new_df.drop(index)
        new_df = pd.concat([new_df, pd.DataFrame(processed_datasets)])
    return new_df


def internal_transform(result, max_duration):
    """
    Transforms the given result into a DataFrame and performs additional processing.

    Args:
        result (dict): The result to be transformed.
        max_duration (int, optional): The maximum duration allowed for a segment. Defaults to 15.

    Returns:
        pandas.DataFrame: The transformed DataFrame.
    """
    df = pd.DataFrame(result["segments"])
    df["segment_length"] = df["end"] - df["start"]
    extras = df[df["segment_length"] > max_duration]
    if extras.empty:
        return df
    else:
        extra_indices = extras.index
        extra_list = extra_indices.tolist()
        new_df = process_extra_indices_internal(df, extra_list, max_duration)
        new_df = new_df.sort_values("start")
        new_df = new_df.reset_index(drop=True)

        return new_df


def external_transform(
    result,
    encoder=HuggingFaceEncoder(name="sentence-transformers/all-MiniLM-L6-v2"),
    max_duration=15,
    min_split_tokens=20,
    max_split_tokens=30,
    window_size=2,
    batch_size=32,
):
    """
    Transforms the given result using external transformations.

    Args:
        result (DataFrame): The input result DataFrame. (Whisper Transcription Result)
        encoder: The encoder instance to be used for creating semantic chunks.
        max_duration (int, optional): The maximum duration of a chunk. Defaults to 15.
        min_split_tokens (int): The minimum number of tokens for a chunk.
        max_split_tokens (int): The maximum number of tokens for a chunk.
        window_size (int): The window size for the chunking algorithm.
        batch_size (int): The batch size for processing chunks.

    Returns:
        DataFrame: The transformed DataFrame.
    """
    internal_transformed_df = internal_transform(result, max_duration)
    semantic_chunks = create_semantic_chunks(
        encoder,
        internal_transformed_df,
        min_split_tokens,
        max_split_tokens,
        window_size,
        batch_size,
    )
    df = transform_chunks(internal_transformed_df, semantic_chunks)
    extras = df[df["chunk_length"] > max_duration]

    if extras.empty:
        df = df.drop(columns=["chunk_id_mapping"])
        return df
    else:
        extra_indices = extras.index
        extra_list = extra_indices.tolist()
        new_df = process_extra_indices_external(df, extra_list, max_duration)
        new_df = new_df.sort_values("start")
        new_df = new_df.reset_index(drop=True)
        new_df = new_df.drop(columns=["chunk_id_mapping", "speaker"])
        new_df.index.name = "chunk_id"
        new_df.reset_index(inplace=True)

        return new_df


def create_semantic_chunks(
    encoder,
    result,
    min_split_tokens=20,
    max_split_tokens=30,
    window_size=2,
    batch_size=32,
):
    """
    Creates semantic chunks from the input text using a specified encoder.

    Parameters:
    - encoder: The encoder to be used for chunking. Example: encoder = HuggingFaceEncoder(name="Snowflake/snowflake-arctic-embed-l")
    - result: The input data containing text to be chunked.
    - min_split_tokens (int): The minimum number of tokens for a chunk.
    - max_split_tokens (int): The maximum number of tokens for a chunk.
    - window_size (int): The window size for the chunking algorithm.
    - batch_size (int): The batch size for processing chunks.

    Returns:
    - semantic_chunks: A list of semantic chunks derived from the input text.
    """
    chunker = StatisticalChunker(
        encoder=encoder,
        dynamic_threshold=True,
        min_split_tokens=min_split_tokens,
        max_split_tokens=max_split_tokens,
        window_size=window_size,
        enable_statistics=False,  # to print chunking stats
        # plot_chunks = True, # Comment out to disable plotting (Cleaner Notebook)
    )

    splits = result["text"].tolist()
    semantic_chunks = chunker._chunk(
        splits=splits, enforce_max_tokens=True, batch_size=batch_size
    )

    return semantic_chunks


def transform_chunks(internal_transformed_df, semantic_chunks):
    """
    Transforms the internal transformed DataFrame by grouping the data based on chunk IDs.

    Args:
        internal_transformed_df (DataFrame): The internal transformed DataFrame.
        semantic_chunks (list): A list of semantic chunks.

    Returns:
        DataFrame: The transformed DataFrame grouped by chunk ID mapping.
    """
    df = internal_transformed_df.copy()
    chunk_id_list = []

    for i in range(len(semantic_chunks)):
        for j in range(len(semantic_chunks[i].splits)):
            chunk_id_list.append(i)

    df["chunk_id_mapping"] = chunk_id_list
    df_grouped = (
        df.groupby("chunk_id_mapping")
        .agg(
            {
                "text": " ".join,
                "start": "min",
                "end": "max",
                "words": lambda x: [i for sublist in x for i in sublist],
            }
        )
        .reset_index()
    )

    df_grouped["chunk_length"] = df_grouped["end"] - df_grouped["start"]

    return df_grouped
