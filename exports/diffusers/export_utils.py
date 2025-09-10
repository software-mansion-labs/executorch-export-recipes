"""
The model exporting to .pte file is implemented according to:  
https://docs.pytorch.org/executorch/stable/using-executorch-export.html#export-and-lowering
"""

import torch
import torch.nn as nn
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import export
import json
import os

def export_model(model, input, model_dir):
    with torch.no_grad():
        exported_program = export(model, input)
        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner = [XnnpackPartitioner()],
        ).to_executorch()

    model_path = os.path.join(model_dir, "model.pte")
    os.makedirs(model_dir, exist_ok=True)
    with open(model_path, "wb") as file:
        file.write(executorch_program.buffer)

def export_tokenizer_config(tokenizer, config_dir):
    config = {
        "version": "1.0",
        "truncation": {
            "max_length": tokenizer.model_max_length,
            "strategy": "LongestFirst",
            "stride": 0,
        },
        "padding": {
            "strategy": "BatchLongest",
            "direction": "Right",
            "pad_to_multiple_of": tokenizer.model_max_length,
            "pad_id": tokenizer.pad_token_id,
            "pad_type_id": 0,
            "pad_token": tokenizer.pad_token,
        },
        "added_tokens": [
            {
                "id": tokenizer.bos_token_id,
                "special": True,
                "content": tokenizer.bos_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
            },
            {
                "id": tokenizer.eos_token_id,
                "special": True,
                "content": tokenizer.eos_token,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": True,
            },
        ],
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": True,
            "handle_chinese_chars": True,
            "strip_accents": None,
            "lowercase": True,
        },
        "pre_tokenizer": {"type": "BertPreTokenizer"},
        "post_processor": {
            "type": "TemplateProcessing",
            "single": [
                {
                    "SpecialToken": {
                    "id": tokenizer.bos_token,
                    "type_id": 0
                    }
                },
                {
                    "Sequence": {
                    "id": "A",
                    "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                    "id": tokenizer.eos_token,
                    "type_id": 0
                    }
                }
            ],
            "pair": [
                {
                    "SpecialToken": {
                    "id": tokenizer.bos_token,
                    "type_id": 0
                    }
                },
                {
                    "Sequence": {
                    "id": "A",
                    "type_id": 0
                    }
                },
                {
                    "SpecialToken": {
                    "id": tokenizer.eos_token,
                    "type_id": 0
                    }
                },
                {
                    "Sequence": {
                    "id": "B",
                    "type_id": 1
                    }
                },
                {
                    "SpecialToken": {
                    "id": tokenizer.eos_token,
                    "type_id": 1
                    }
                }
            ],
            "special_tokens": {
                "bos_token": {
                    "id": tokenizer.bos_token,
                    "ids": [tokenizer.bos_token_id],
                    "tokens": [tokenizer.bos_token],
                },
                "eos_token": {
                    "id": tokenizer.eos_token,
                    "ids": [tokenizer.eos_token_id],
                    "tokens": [tokenizer.eos_token],
                },
                "unk_token": {
                    "id": tokenizer.unk_token,
                    "ids": [tokenizer.unk_token_id],
                    "tokens": [tokenizer.unk_token],
                },
            },
        },
        "decoder": {
            "type": "WordPiece",
            "prefix": "",
            "add_prefix_space": False,
            "cleanup": True,
        },
        "model": {
            "type": "WordPiece",
            "unk_token": tokenizer.unk_token or "unk_token",
            "continuing_subword_prefix": "",
            "max_input_chars_per_word": 100,
            "vocab": tokenizer.get_vocab(),
        },
    }
    config_path = os.path.join(config_dir, "tokenizer.json")
    with open(config_path, "w") as file:
        json.dump(config, file)


def export_tokenizer(tokenizer, model_dir):
    tokenizer_path = os.path.join(model_dir, "tokenizer")
    tokenizer.save_pretrained(tokenizer_path)
    export_tokenizer_config(tokenizer, tokenizer_path)


def export_scheduler(scheduler, model_dir):
    scheduler_path = os.path.join(model_dir, "scheduler")
    scheduler.save_pretrained(scheduler_path)


def export_encoder(model, input, model_dir):
    export_model(model, input, model_dir)
    config_path = os.path.join(model_dir, "config.json")
    model.config.to_json_file(config_path)


def export_unet(model, input, model_dir):
    export_model(model, input, model_dir)
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "w") as file:
        json.dump(model.config, file)


def export_vae(vae, input, model_dir):

    class VaeDecoderWrapper(nn.Module):
        def __init__(self, vae: nn.Module):
            super().__init__()
            self.vae = vae

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            output = self.vae.decode(input)[0]

            if output.shape[0] == 1:
                output = output.squeeze(0)
            output = output.permute(1, 2, 0)
            output = output / 2 + 0.5
            output = torch.clamp(output, 0.0, 1.0)
            output = output * 255

            return output
    
    vae_wrapper = VaeDecoderWrapper(vae)
    export_model(vae_wrapper, input, model_dir)