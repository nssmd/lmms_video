import json
import os
import pathlib
import random
import shutil
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
import yaml

import lmms_engine.parallel.process_group_manager as pgm
from lmms_engine.mapping_func import (
    DATASET_MAPPING,
    create_model_from_config,
    create_model_from_pretrained,
)
# Import datasets to register them
from lmms_engine.datasets import TextDataset, VisionSFTDataset, VisionAudioSFTDataset, FinewebEduDataset
from lmms_engine.models.utils import setup_flops_counter
from lmms_engine.parallel.sequence_parallel.ulysses import (
    set_ulysses_sequence_parallel_group,
)

from ..models.monkey_patch import CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN
from ..models.monkey_patch import (
    _apply_liger_kernel_to_instance as _apply_liger_kernel_to_custom_instance,
)
from ..utils import Logging
from ..utils.train_utils import TrainUtilities
from .config import TrainerConfig
from .dllm_trainer import DLLMTrainer
from .fsdp2_trainer import FSDP2SFTTrainer
from .trainer import Trainer

# from transformers import Trainer


class TrainRunner:
    """
    This is a base train runner to wrap all other trainer or your training logic
    """

    def __init__(self, config: TrainerConfig) -> None:
        self.set_random_seed()
        self.train_dataset_config = config.dataset_config
        if config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset_config = deepcopy(config.dataset_config)
            # Never use packing for eval dataset
            self.eval_dataset_config.packing = False
            self.eval_dataset_config.dataset_path = (
                config.dataset_config.eval_dataset_path
            )
        self.model_config = config.model_config
        self.config = config

    def build(self):
        self.create_sp_dis_group()
        self.model = self._build_model()
        if self.config.dataset_config.eval_dataset_path is not None:
            self.eval_dataset = self._build_eval_dataset()
        else:
            self.eval_dataset = None
        self.train_dataset = self._build_train_dataset()
        if self.model_config.pretrain_mm_mlp_adapter is not None:
            self._load_mm_projector()
        if self.config.trainer_args.use_liger_kernel:
            self._apply_liger_kernel()
            # Set to False as we already apply the liger kernel by ourselves
            self.config.trainer_args.use_liger_kernel = False
        # Setup autoregressive module if enabled
        self._setup_autoregressive_module()
        self.trainer = self._build_trainer()

    def _build_model(self):
        load_from_pretrained_path = self.model_config.load_from_pretrained_path
        load_from_config = self.model_config.load_from_config
        if load_from_pretrained_path is not None:
            model_class = create_model_from_pretrained(load_from_pretrained_path)

            # Directly load the model without config conversion
            # The model weights already have the correct config embedded
            model = model_class.from_pretrained(
                load_from_pretrained_path,
                attn_implementation=self.model_config.attn_implementation,
                torch_dtype=(torch.bfloat16 if self.config.trainer_args.bf16 else None),
                trust_remote_code=True,
            )
        elif load_from_config is not None:
            model_type = load_from_config.get("model_type", None)
            # Handle both nested and flat config structures
            init_config = load_from_config.get("config", None)
            if init_config is None:
                # If no nested config, use the load_from_config dict directly (excluding model_type)
                init_config = {
                    k: v for k, v in load_from_config.items() if k != "model_type"
                }
            model_class, m_config = create_model_from_config(model_type, init_config)
            model = model_class.from_config(m_config)
        else:
            raise ValueError(
                "No model name or pretrained path provided. Please provide one of them."
            )

        if self.model_config.overwrite_config:
            for key, value in self.model_config.overwrite_config.items():
                setattr(model.config, key, value)
                Logging.info(f"Overwrite {key} to {value}")

        setup_flops_counter(model.config)
        Logging.info(f"Model Structure: {model}")
        Logging.info(
            f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9} B"
        )
        return model

    def _apply_liger_kernel(self):
        kwargs = {"use_rmpad": self.config.trainer_args.use_rmpad}
        try:
            from liger_kernel.transformers import _apply_liger_kernel_to_instance
            from liger_kernel.transformers.monkey_patch import (
                MODEL_TYPE_TO_APPLY_LIGER_FN,
            )
        except ImportError as e:
            Logging.error(
                "You have set `use_liger_kernel` to `True` but liger-kernel >= 0.3.0 is not available. "
                "Please install it with `pip install liger-kernel`"
            )

        model_type = getattr(self.model, "config", None) and getattr(
            self.model.config, "model_type", None
        )
        if model_type in CUSTOM_MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_custom_instance(self.model, **kwargs)
        # If the model itself is already in liger kernel,
        # we should not apply the liger kernel again
        elif model_type in MODEL_TYPE_TO_APPLY_LIGER_FN:
            Logging.info(f"Try to apply liger kernel on the model {model_type}")
            _apply_liger_kernel_to_instance(self.model)
        else:
            Logging.info(
                f"Try to apply custom liger kernel on the language model of the model {model_type}"
            )
            try:
                _apply_liger_kernel_to_custom_instance(
                    self.model.language_model, **kwargs
                )
                Logging.info(
                    f"Successfully apply custom liger kernel on the language model of the model {model_type}"
                )
                return
            except Exception as e:
                Logging.error(
                    f"Try to apply custom liger kernel on the language model of the model {model_type}, but failed with exceptions : \n {e}"
                )

            try:
                _apply_liger_kernel_to_instance(self.model.language_model)
                Logging.info(
                    f"Successfully apply liger kernel on the language model of the model {model_type}"
                )
                return
            except Exception as e:
                Logging.error(
                    f"Try to apply liger kernel on the language model of the model {model_type}, but failed with exceptions : \n {e}"
                )

    def _setup_autoregressive_module(self):
        """Setup autoregressive reconstruction module if enabled"""
        if not self.model_config.overwrite_config:
            return

        enable_ar = self.model_config.overwrite_config.get("enable_autoregressive", False)
        if not enable_ar:
            return

        Logging.info("🔧 Setting up autoregressive reconstruction module")

        # Get autoregressive config
        ar_config = self.model_config.overwrite_config.get("autoregressive_config", {})

        # Import and create autoregressive module
        from ..models.autoregressive_reconstruction import (
            create_autoregressive_reconstruction_module
        )

        # Get model's hidden_size (LLM dimension)
        if hasattr(self.model.config, 'text_config'):
            hidden_size = self.model.config.text_config.hidden_size
        elif hasattr(self.model.config, 'hidden_size'):
            hidden_size = self.model.config.hidden_size
        else:
            raise ValueError("Cannot determine model hidden_size for autoregressive module")

        # Get vision tower
        if hasattr(self.model, 'vision_tower'):
            vision_tower = self.model.vision_tower
        elif hasattr(self.model, 'get_vision_tower'):
            vision_tower = self.model.get_vision_tower()
        else:
            raise ValueError("Cannot find vision_tower in model")

        # Create autoregressive module
        self.model.autoregressive_module = create_autoregressive_reconstruction_module(
            vision_tower=vision_tower,
            hidden_size=hidden_size,
            config=ar_config
        )

        Logging.info(f"✅ Autoregressive module created with hidden_size={hidden_size}")

        # Monkey-patch the forward method to add autoregressive loss
        original_forward = self.model.forward

        def forward_with_autoregressive(*args, **kwargs):
            # Extract video_frames if present
            video_frames = kwargs.pop('video_frames', None)

            # Call original forward
            outputs = original_forward(*args, **kwargs)

            # Add autoregressive loss if training and video_frames provided
            if self.model.training and video_frames is not None and hasattr(outputs, 'loss'):
                # Get hidden states (need output_hidden_states=True in config)
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    last_hidden_state = outputs.hidden_states[-1]

                    # Compute autoregressive loss
                    ar_loss = self.model.autoregressive_module.compute_autoregressive_loss(
                        last_hidden_state, video_frames
                    )

                    # Add to main loss
                    outputs.loss = outputs.loss + ar_loss

            return outputs

        self.model.forward = forward_with_autoregressive
        Logging.info("✅ Model forward method patched with autoregressive loss")

    def _load_mm_projector(self):
        pretrain_mm_mlp_adapter = self.config.model_config.pretrain_mm_mlp_adapter
        mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

        def get_w(weights, keyword):
            return {
                k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k
            }

        deepspeed3_enabled = hasattr(
            [p for p in self.model.multi_modal_projector.parameters()][0], "ds_id"
        )

        TrainUtilities.load_zero_partitions(
            self.model.multi_modal_projector,
            get_w(mm_projector_weights, "multi_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )
        TrainUtilities.load_zero_partitions(
            self.model.audio_modal_projector,
            get_w(mm_projector_weights, "audio_modal_projector"),
            deepspeed3_enabled,
            pretrain_mm_mlp_adapter,
        )

        Logging.info(
            f"Loaded multi_modal_projector,audio_modal_projector weights from {pretrain_mm_mlp_adapter}."
        )

    def _build_train_dataset(self):
        dataset_cls = DATASET_MAPPING[self.train_dataset_config.dataset_type]
        dataset = dataset_cls(self.train_dataset_config)
        dataset.build()
        return dataset

    def _build_eval_dataset(self):
        dataset_cls = DATASET_MAPPING[self.eval_dataset_config.dataset_type]
        dataset = dataset_cls(self.eval_dataset_config)
        dataset.build()
        return dataset

    def save_config(self):
        output_dir = self.config.trainer_args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/training_config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        if self.config.dataset_config.dataset_format == "yaml":
            if self.config.dataset_config.dataset_path:
                # Copy the external yaml to output dir
                yaml_path = self.config.dataset_config.dataset_path
                shutil.copy(yaml_path, f"{output_dir}/dataset.yaml")
            elif self.config.dataset_config.datasets:
                # For inline datasets, save them to a yaml file
                with open(f"{output_dir}/dataset.yaml", "w") as f:
                    yaml.dump({"datasets": self.config.dataset_config.datasets}, f)

    def set_random_seed(self, random_seed: int = 42):
        # Setting random seed for all
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        Logging.info(f"Set random seed to {random_seed}")
        return random_seed

    def create_sp_dis_group(self):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        sp_ulysses_degree = self.config.trainer_args.sp_ulysses_degree
        sp_degree = sp_ulysses_degree * 1  # ring attn always 1, kept for clarity

        total_group_size = sp_degree
        assert (
            world_size % total_group_size == 0
        ), f"world_size={world_size} must be divisible by total_group_size={total_group_size}"

        set_ulysses_sequence_parallel_group(pgm.process_group_manager.cp_group)

    def _build_trainer(self):
        if self.config.trainer_type == "hf_trainer":
            trainer_cls = Trainer
        elif self.config.trainer_type == "fsdp2_trainer":
            trainer_cls = FSDP2SFTTrainer
        elif self.config.trainer_type == "dllm_trainer":
            trainer_cls = DLLMTrainer
        else:
            raise ValueError(
                f"Unsupported trainer type: {self.config.trainer_type}"
            )
        from transformers.trainer_pt_utils import AcceleratorConfig

        trainer = trainer_cls(
            model=self.model,
            args=self.config.trainer_args,
            data_collator=self.train_dataset.get_collator(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.train_dataset.processor,
        )
        return trainer

    def run(self, **kwargs):
        self.save_config()
        if self.config.trainer_args.freeze_modules:
            for modules in self.config.trainer_args.freeze_modules:
                cls = getattr(self.model, modules, None)
                if cls is not None:
                    for param in cls.parameters():
                        param.requires_grad = False

        if list(pathlib.Path(self.config.trainer_args.output_dir).glob("checkpoint-*")):
            self.trainer.train(resume_from_checkpoint=True)
        else:
            self.trainer.train()
        # Save the state for hf_trainer
        if hasattr(self.trainer, "save_state"):
            self.trainer.save_state()
            self.safe_save_model_for_hf_trainer(
                self.trainer, self.config.trainer_args.output_dir
            )

    def safe_save_model_for_hf_trainer(self, trainer: Trainer, output_dir: str):
        """Collects the state dict and dump to disk."""
        trainer.accelerator.wait_for_everyone()
        torch.cuda.synchronize()
        check_only_save_mm_adapter = self.config.trainer_args.only_save_mm_adapter
        Logging.info(f"Only save projectors: {check_only_save_mm_adapter}")

        if check_only_save_mm_adapter:
            # Only save Adapter
            keys_to_match = ["multi_modal_projector", "audio_modal_projector"]

            weight_to_save = TrainUtilities.get_mm_adapter_state_maybe_zero_3(
                trainer.model.named_parameters(), keys_to_match
            )
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith("checkpoint-"):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(
                        weight_to_save,
                        os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                    )
                else:
                    torch.save(
                        weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                    )
            return
        if trainer.deepspeed:
            trainer.save_model(output_dir)
            return
        if self.config.trainer_args.fsdp2:
            # For fsdp we merge the shards into a single checkpoint after the training is done
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(output_dir)
            return

        state_dict = trainer.model.state_dict()
        if trainer.args.should_save:
            cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
