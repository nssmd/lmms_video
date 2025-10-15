import os
from copy import deepcopy
from typing import Dict

import torch
from PIL import Image

from lmms_engine.mapping_func import register_dataset

from ..utils.train_utils import TrainUtilities
from .base_dataset import BaseDataset
from .collator import VisionCollator


@register_dataset("vision")
class VisionSFTDataset(BaseDataset):
    def load_from_csv(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        """Load from CSV data directly without intermediate transformation."""
        images_list = []
        videos = []
        kwargs = {}

        # Build messages directly from CSV data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": data["video"]}},
                    {"type": "text", "text": data["prompt"]},
                ],
            }
        ]

        # Process video content directly
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(
            images=images, hf_messages=hf_messages, videos=videos, **kwargs
        )
        return inputs

    def load_from_json(self, data, data_folder=None) -> Dict[str, torch.Tensor]:
        # TODO Write a protocol for vision openai input
        images_list = []
        videos = []
        kwargs = {}
        messages = data["messages"]
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    # Loading videos with fps
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    # Update kwargs
                    kwargs["fps"] = sample_fps

        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        if data_folder is not None:
            images = [
                Image.open(os.path.join(data_folder, image)) for image in images_list
            ]
        else:
            images = [Image.open(image) for image in images_list]
        if len(images) == 0:
            images = None
        if len(videos) == 0:
            videos = None
        inputs = self.processor.process(
            images=images, hf_messages=hf_messages, videos=videos, **kwargs
        )
        return inputs

    def load_from_hf(self, data) -> Dict[str, torch.Tensor]:
        # Similar to load_from_json, but data comes from HF dataset
        images_list = []
        videos = []
        kwargs = {}
        messages = data["messages"]
        
        # Get data_folder from config (for parquet datasets with relative paths)
        data_folder = self.config.data_folder if hasattr(self.config, 'data_folder') else None
        
        # Extract videos from messages (same as load_from_json)
        for message in messages:
            for content in message["content"]:
                if content["type"] == "image_url":
                    images_list.append(content["image_url"]["url"])
                elif content["type"] == "video_url":
                    # Load videos with fps and data_folder for relative paths
                    frames, sample_fps = self.load_videos(
                        content["video_url"]["url"],
                        data_folder=data_folder,
                        fps=self.config.fps,
                    )
                    videos.append(frames)
                    kwargs["fps"] = sample_fps
        
        hf_messages = TrainUtilities.convert_open_to_hf(messages)
        
        # Fallback: use data["image"] field if no images/videos in messages
        if len(images_list) == 0 and "image" in data and data["image"] is not None:
            if isinstance(data["image"], list):
                images = data["image"]
            else:
                images = [data["image"]]
        else:
            # Load images with data_folder support
            if len(images_list) > 0:
                if data_folder is not None:
                    images = [Image.open(os.path.join(data_folder, img)) for img in images_list]
                else:
                    images = [Image.open(img) for img in images_list]
            else:
                images = None
        
        if len(videos) == 0:
            videos = None
        
        inputs = self.processor.process(images=images, hf_messages=hf_messages, videos=videos, **kwargs)
        return inputs

    def get_collator(self):
        return VisionCollator(self.processor)
