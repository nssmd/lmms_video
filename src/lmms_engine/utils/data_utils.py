import json
import math
from io import BytesIO
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Literal, Tuple, Union

import jsonlines
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, concatenate_datasets, load_from_disk
from librosa import resample
from tqdm import tqdm

from .logging_utils import Logging
from .train_utils import TrainUtilities

FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


class DataUtilities:
    @staticmethod
    def load_json(path: str) -> List[Dict[str, List]]:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    @staticmethod
    def load_jsonlines(path: str) -> List[Dict[str, List]]:
        data_list = []
        with jsonlines.open(path, "r") as f:
            for data in f:
                data_list.append(data)

        return data_list

    @staticmethod
    def load_csv(path: str) -> List[Dict[str, str]]:
        """Load CSV file and convert to list of dictionaries."""
        df = pd.read_csv(path)
        # Convert DataFrame to list of dictionaries
        data_list = df.to_dict("records")
        return data_list

    @staticmethod
    def maybe_load_json_or_jsonlines_or_csv(
        path: str, data_type: Literal["json", "jsonl", "csv"]
    ) -> List[Dict[str, List]]:
        if data_type == "json":
            return DataUtilities.load_json(path)
        elif data_type == "jsonl":
            return DataUtilities.load_jsonlines(path)
        elif data_type == "csv":
            return DataUtilities.load_csv(path)
        else:
            raise NotImplementedError

    @staticmethod
    def maybe_load_by_type(
        path: str, data_type: Literal["json", "jsonl", "csv", "arrow", "txt"]
    ) -> Union[List[Dict[str, List]], Dataset]:
        if data_type == "arrow":
            dataset = load_from_disk(path)
        elif data_type == "parquet":
            dataset = Dataset.from_parquet(path)
        elif data_type == "txt":
            # 处理txt文件 - 读取路径列表并加载对应的JSON数据
            dataset = DataUtilities.load_subset_from_txt(path)
        else:
            dataset = DataUtilities.maybe_load_json_or_jsonlines_or_csv(path, data_type)

        # Force to load in Dataset format if load in yaml
        # For better streaming data
        if not isinstance(dataset, Dataset):
            dataset = Dataset.from_list(dataset)
        return dataset

    @staticmethod
    def wrap_func(args):
        path, data_type = args
        return DataUtilities.maybe_load_by_type(path, data_type)

    @staticmethod
    def load_yaml(path: str) -> Tuple[List[Dict[str, List]], List[str]]:
        data_list = []
        data_folder_list = []
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
            datasets = yaml_data.get("datasets")
            data_paths = [dataset.get("path") for dataset in datasets]
            data_folders = [dataset.get("data_folder") for dataset in datasets]
            data_types = [dataset.get("data_type") for dataset in datasets]
            with Pool(cpu_count()) as p:
                Logging.info("Loading data with multiprocess...")
                nested_data_list = list(
                    p.imap(DataUtilities.wrap_func, zip(data_paths, data_types))
                )

            for data, data_folder, data_path in zip(
                nested_data_list, data_folders, data_paths
            ):
                Logging.info(f"Data : {data_path}")
                if isinstance(data, Dataset):
                    data_list.append(data)
                else:
                    Logging.info(f"Convert to hf dataset")
                    data = Dataset.from_list(data)
                    data_list.append(data)
                Logging.info(f"Dataset size: {len(data)}")
                data_folder_list.extend([data_folder] * len(data))
            data_list = concatenate_datasets(data_list)
        return data_list, data_folder_list

    @staticmethod
    def smart_nframes(
        total_frames: int,
        video_fps: int | float,
        fps: int,
    ) -> int:
        """calculate the number of frames for video used for model inputs.

        Args:
            ele (dict): a dict contains the configuration of video.
                support either `fps` or `nframes`:
                    - nframes: the number of frames to extract for model inputs.
                    - fps: the fps to extract frames for model inputs.
                        - min_frames: the minimum number of frames of the video, only used when fps is provided.
                        - max_frames: the maximum number of frames of the video, only used when fps is provided.
            total_frames (int): the original total number of frames of the video.
            video_fps (int | float): the original fps of the video.

        Raises:
            ValueError: nframes should in interval [FRAME_FACTOR, total_frames].

        Returns:
            int: the number of frames for video used for model inputs.
        """
        min_frames = DataUtilities.ceil_by_factor(FPS_MIN_FRAMES, FRAME_FACTOR)
        max_frames = DataUtilities.floor_by_factor(
            min(FPS_MAX_FRAMES, total_frames), FRAME_FACTOR
        )
        nframes = total_frames / video_fps * fps
        nframes = min(min(max(nframes, min_frames), max_frames), total_frames)
        nframes = DataUtilities.floor_by_factor(nframes, FRAME_FACTOR)
        if not (FRAME_FACTOR <= nframes and nframes <= total_frames):
            raise ValueError(
                f"nframes should in interval [{FRAME_FACTOR}, {total_frames}], but got {nframes}."
            )
        return nframes

    @staticmethod
    def round_by_factor(number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor

    @staticmethod
    def ceil_by_factor(number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor

    @staticmethod
    def floor_by_factor(number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor

    @staticmethod
    def download_blob_to_stream(
        storage_client,
        bucket_name: str,
        source_blob_name: str,
        file_obj: BytesIO,
        storage_type: Literal["gcs", "azure"] = "azure",
        max_retries: int = 5,
    ) -> BytesIO:
        for i in range(max_retries):
            try:
                if storage_type == "gcs":
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(source_blob_name)
                    blob.download_to_file(file_obj)
                elif storage_type == "azure":
                    blob_client = storage_client.get_blob_client(
                        container=bucket_name, blob=source_blob_name
                    )
                    blob_client.download_blob().readinto(file_obj)
                break
            except Exception as e:
                Logging.error(f"Attempt {i} Error downloading blob: {source_blob_name}")
                Logging.error(f"Error: {e}")
                Logging.error(f"Retrying ...")

        return file_obj

    @staticmethod
    def resample_audio(
        audio_array: np.ndarray, original_sr: int, target_sr: int
    ) -> np.ndarray:
        audio_resample_array = resample(
            audio_array, orig_sr=original_sr, target_sr=target_sr
        )
        return audio_resample_array

    @staticmethod
    def load_subset_from_txt(txt_file_path: str) -> Dataset:
        """从txt文件中读取文件路径列表，并加载所有数据（支持JSON/JSONL/Parquet）
        
        Args:
            txt_file_path: 包含文件路径列表的txt文件路径
            
        Returns:
            合并后的Dataset对象
        """
        import os
        import json
        from datasets import Dataset
        import pandas as pd
        
        Logging.info(f"从txt文件加载数据子集: {txt_file_path}")
        
        # 读取文件路径列表
        file_paths = []
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释行和空行
                if line and not line.startswith('#'):
                    file_paths.append(line)
        
        Logging.info(f"找到 {len(file_paths)} 个文件")
        
        all_data = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                Logging.warning(f"文件不存在，跳过: {file_path}")
                continue
                
            Logging.info(f"加载文件: {os.path.basename(file_path)}")
            
            try:
                # 检查文件类型
                if file_path.endswith('.parquet'):
                    # 加载Parquet文件
                    df = pd.read_parquet(file_path)
                    data_list = df.to_dict('records')
                    all_data.extend(data_list)
                elif file_path.endswith('.jsonl'):
                    # 加载JSONL文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    data = json.loads(line)
                                    all_data.append(data)
                                except json.JSONDecodeError as e:
                                    Logging.warning(f"  JSONL解析错误: {e}")
                                    continue
                elif file_path.endswith('.json'):
                    # 加载JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        try:
                            data = json.load(f)
                            if isinstance(data, list):
                                all_data.extend(data)
                            else:
                                all_data.append(data)
                        except json.JSONDecodeError as e:
                            Logging.warning(f"  JSON解析错误: {e}")
                            continue
                else:
                    Logging.warning(f"  不支持的文件格式: {file_path}")
                    continue
                                
            except Exception as e:
                Logging.error(f"加载文件失败 {file_path}: {e}")
                continue
        
        Logging.info(f"总共加载了 {len(all_data)} 条数据")
        return Dataset.from_list(all_data)

    @staticmethod
    def load_inline_datasets(
        datasets: List[Dict],
    ) -> Tuple[List[Dict[str, List]], List[str]]:
        """Load datasets from inline configuration (similar to load_yaml but without file loading).

        Args:
            datasets: List of dataset configurations with path, data_folder, and data_type

        Returns:
            Tuple of (data_list, data_folder_list)
        """
        data_list = []
        data_folder_list = []

        if not datasets:
            return data_list, data_folder_list

        data_paths = [dataset.get("path") for dataset in datasets]
        data_folders = [dataset.get("data_folder", "") for dataset in datasets]
        data_types = [dataset.get("data_type", "json") for dataset in datasets]

        with Pool(cpu_count()) as p:
            Logging.info("Loading data with multiprocess...")
            nested_data_list = list(
                p.imap(DataUtilities.wrap_func, zip(data_paths, data_types))
            )

        for data, data_folder, data_path in zip(
            nested_data_list, data_folders, data_paths
        ):
            Logging.info(f"Data : {data_path}")
            if isinstance(data, Dataset):
                data_list.append(data)
            else:
                Logging.info(f"Convert to hf dataset")
                data = Dataset.from_list(data)
                data_list.append(data)
            Logging.info(f"Dataset size: {len(data)}")
            data_folder_list.extend([data_folder] * len(data))
        data_list = concatenate_datasets(data_list)

        return data_list, data_folder_list
