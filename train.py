import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import HfApi, create_repo
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import (
    CustomDiffusionXFormersAttnProcessor,
)
from utils.CustomAttnProcessor import CustomDiffusionAttnProcessor
from utils.CustomModelLoader import CustomModelLoader
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import wandb

from utils.CrossAttnMap import AttentionStore, aggregate_current_attention

from skimage.metrics import structural_similarity as ssim


torch.autograd.set_detect_anomaly(True)

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

logger = get_logger(__name__)

def save_tensor_to_image(tensor, filename):
    # First, we need to denormalize the tensor and convert it back to a PIL Image
    transform = transforms.Compose([
        transforms.Normalize(mean=[-0.5/0.5], std=[1/0.5]),  # This undoes the normalization
        transforms.ToPILImage()  # This converts the tensor to a PIL Image
    ])
    image = transform(tensor)

    # Then, we can save the image to the local folder
    image.save(filename)

def find_the_identify_token_index(tokens, modifier_token_id):
    """
    The structure of the prompt is like: A photo of ... and ...
    So we need to find the index of the first "and" token, which is the token before the second object.
    """

    modifier_token_sets = [] # this is to record the modifier token with its descriptions
    target_indices = [token_id for token_id, token in enumerate(tokens) if token in modifier_token_id]
    valid_length = torch.where(tokens == 49407)[0] # 49407 is the token id of EoT


    of_index = [token_id for token_id, token in enumerate(tokens) if token == 539] # 539 is the token id of "of"
    assert len(of_index) == 1, "There should be only one 'of' token in the prompt."
    and_index = [token_id for token_id, token in enumerate(tokens) if token == 537] # 537 is the token id of "and"

    if len(and_index) != 0:
        modifier_token_set = [token_id for token_id in range(of_index[0]+1, and_index[0])]
        modifier_token_sets.append(modifier_token_set)
        if len(and_index) > 1:
            for and_index_id in range(1, len(and_index)):
                modifier_token_set = [token_id for token_id in range(and_index[and_index_id]+1, and_index[and_index_id+1])]
                modifier_token_sets.append(modifier_token_set)
        modifier_token_set = [token_id for token_id in range(and_index[-1]+1, valid_length)]
        modifier_token_sets.append(modifier_token_set)
    else:
        modifier_token_set = [token_id for token_id in range(of_index[0]+1, valid_length)]
        modifier_token_sets.append(modifier_token_set)

    assert len(modifier_token_sets) == len(target_indices), "The number of modifier tokens should be the same as the number of target indices."

    #check result
    for target_index_id, target_index in enumerate(target_indices):
        if target_index not in modifier_token_sets[target_index_id]:
            raise ValueError("The target index is not in the modifier token set.")


    target_indices.append(0) # add the SoT token
    modifier_token_sets.append([0])
    return target_indices, modifier_token_sets

def show_attention_map_during_training(cross_attention_map, obj):
    split_tensors = torch.split(cross_attention_map, 1, dim=0)

    # create a list to store the PIL images
    images = []

    # loop over the split tensors and show the PIL image of each element
    for i, tensor in enumerate(split_tensors):
        # convert the tensor to a PIL image
        image = Image.fromarray(tensor.squeeze().mul(255).clamp(0, 255).byte().cpu().numpy())

        # append the image to the list
        images.append(image)

    # combine the images into a 1 row 4 column image
    combined_image = Image.new(mode="RGB", size=(cross_attention_map.shape[1] * len(split_tensors), cross_attention_map.shape[2]))
    for i, image in enumerate(images):
        combined_image.paste(image, (i * cross_attention_map.shape[1], 0))

    # log the combined image to wandb
    wandb.log({f"{obj}": wandb.Image(combined_image)})


def freeze_params(params):
    for param in params:
        param.requires_grad = False


def save_model_card(repo_id: str, images=None, base_model=str, prompt=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- custom-diffusion
inference: true
---
    """
    model_card = f"""
# Custom Diffusion - {repo_id}

These are Custom Diffusion adaption weights for {base_model}. The weights were trained on {prompt} using [Custom Diffusion](https://www.cs.cmu.edu/~custom-diffusion). You can find some example images in the following. \n
{img_str}

\nFor more details on the training, please follow [this link](https://github.com/huggingface/diffusers/blob/main/examples/custom_diffusion).
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    token_ids = [example["token_ids"] for example in examples]
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids_1"] for example in examples]
        pixel_values += [example["class_images_1"] for example in examples]
        mask += [example["class_mask_1"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.stack(pixel_values)
    mask = torch.stack(mask)

    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {"input_ids": input_ids, "pixel_values": pixel_values, "mask": mask.unsqueeze(1), "token_ids": token_ids}
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class CustomDiffusionDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        mask_size=64,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        aug=True,
        sample_mode = "all",
        object_list = "<new1>+<new2>",
    ):
        self.size = size
        self.mask_size = mask_size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = Image.BILINEAR
        self.aug = aug

        self.object_list = object_list.split("+")

        self.sample_mode = sample_mode

        self.instance_images_path = []
        self.class_images_path_1 = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            inst_img_path = [
                (x, ) for x in Path(concept["instance_data_dir"]).iterdir() if x.is_file()
            ]
            self.instance_images_path.extend(inst_img_path)

            if with_prior_preservation:
                class_data_root_1 = Path(concept["class_data_dir_1"])
                if os.path.isdir(class_data_root_1):
                    class_images_path_1 = list(class_data_root_1.iterdir())
                    class_prompt_1 = [concept["class_prompt_1"] for _ in range(len(class_images_path_1))]
                else:
                    with open(class_data_root_1, "r") as f:
                        class_images_path_1 = f.read().splitlines()
                    with open(concept["class_prompt_1"], "r") as f:
                        class_prompt_1 = f.read().splitlines()

                class_img_path_1 = list(zip(class_images_path_1, class_prompt_1))
                self.class_images_path_1.extend(class_img_path_1[:num_class_images])

        #random.shuffle(self.instance_images_path)
        self.num_instance_images = len(self.instance_images_path)
        self.num_class_images_1 = len(self.class_images_path_1)
        self._length = max(self.num_class_images_1, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.heatmap_transforms = transforms.Compose(
            [  
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(256) if center_crop else transforms.RandomCrop(256),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def preprocess(self, image, scale, resample):
        outer, inner = self.size, scale
        factor = self.size // self.mask_size
        if scale > self.size:
            outer, inner = scale, self.size
        top, left = np.random.randint(0, outer - inner + 1), np.random.randint(0, outer - inner + 1)
        image = image.resize((scale, scale), resample=resample)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
        mask = np.zeros((self.size // factor, self.size // factor))
        if scale > self.size:
            instance_image = image[top : top + inner, left : left + inner, :]
            mask = np.ones((self.size // factor, self.size // factor))
        else:
            instance_image[top : top + inner, left : left + inner, :] = image
            mask[
                top // factor + 1 : (top + scale) // factor - 1, left // factor + 1 : (left + scale) // factor - 1
            ] = 1.0
        return instance_image, mask

    def __getitem__(self, index):
        example = {}
        instance_image = self.instance_images_path[index % self.num_instance_images][0]
        instance_image = Image.open(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        if self.sample_mode == "all":
            instance_prompt = "a photo of " + " and ".join(self.object_list)
            tokens_ids_to_use = list(range(len(self.object_list)))
        elif self.sample_mode == "subset":
            num_determiner = random.random()
            if num_determiner < 1/3:
                tokens_ids_to_use = [0]
            elif 1/3 <= num_determiner < 2/3:
                tokens_ids_to_use = [1]
            else:
                tokens_ids_to_use = random.sample(range(len(self.object_list)), k=2)

            tokens_to_use = [self.object_list[tkn_i] for tkn_i in tokens_ids_to_use]
            instance_prompt = "a photo of " + " and ".join(tokens_to_use)
        

        # apply resize augmentation and create a valid image region mask
        random_scale = self.size
        if self.aug:
            random_scale = (
                np.random.randint(self.size // 3, self.size + 1)
                if np.random.uniform() < 0.66
                else np.random.randint(int(1.2 * self.size), int(1.4 * self.size))
            )
        instance_image, mask = self.preprocess(instance_image, random_scale, self.interpolation)

        if random_scale < 0.6 * self.size:
            instance_prompt = np.random.choice(["a far away ", "very small "]) + instance_prompt
        elif random_scale > self.size:
            instance_prompt = np.random.choice(["zoomed in ", "close up "]) + instance_prompt

        example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
        example["mask"] = torch.from_numpy(mask)
        example["instance_prompt_ids"] = self.tokenizer(
            instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids
        example["token_ids"] = tokens_ids_to_use

        if self.with_prior_preservation:
            class_image_1, class_prompt_1 = self.class_images_path_1[index % self.num_class_images_1]
            class_image_1 = Image.open(class_image_1)
            if not class_image_1.mode == "RGB":
                class_image_1 = class_image_1.convert("RGB")
            example["class_images_1"] = self.image_transforms(class_image_1)
            example["class_mask_1"] = torch.ones_like(example["mask"])
            example["class_prompt_ids_1"] = self.tokenizer(
                class_prompt_1,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

        return example


def save_new_embed(text_encoder, modifier_token_id, accelerator, args, output_dir, safe_serialization=True):
    """Saves the new token embeddings from the text encoder."""
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight
    for x, y in zip(modifier_token_id, args.modifier_token):
        learned_embeds_dict = {}
        learned_embeds_dict[y] = learned_embeds[x]
        filename = f"{output_dir}/{y}.bin"

        if safe_serialization:
            safetensors.torch.save_file(learned_embeds_dict, filename, metadata={"format": "pt"})
        else:
            torch.save(learned_embeds_dict, filename)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Custom Diffusion training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir_1",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt_1",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="custom-diffusion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--train_k",
        action="store_true",
        default=False,
        help="Whether to train the k layer of the cross attention.",
    )
    parser.add_argument(
        "--train_v",
        action="store_true",
        default=False,
        help="Whether to train the v layer of the cross attention.",
    )
    parser.add_argument(
        "--train_q",
        action="store_true",
        default=False,
        help="Whether to train the q layer of the cross attention.",
    )
    parser.add_argument(
        "--train_out",
        action="store_true",
        default=False,
        help="Whether to train the out layer of the cross attention.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=1, help="Number of steps for the warmup in the first phase."
    )
    parser.add_argument(
        "--ssim_threshold", type=float, default=0.8, help="ssim_threshold for updataing masks."
    )
    parser.add_argument(
        "--object_list",
        type=str,
        default=None,
        help="object_list",
    )
    parser.add_argument(
        "--attention_threshold", type=float, default=0.1, help="attention threshold for initilizing masks."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None:
            if args.class_data_dir_1 is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt_1 is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir_1 is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt_1 is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("AttenCraft", config=vars(args))

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    if args.concepts_list is None:
        args.concepts_list = [
            {
                "instance_prompt": args.instance_prompt,
                "class_prompt_1": args.class_prompt_1,
                "instance_data_dir": args.instance_data_dir,
                "class_data_dir_1": args.class_data_dir_1,
            }
        ]
    else:
        with open(args.concepts_list, "r") as f:
            args.concepts_list = json.load(f)

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir_1 = Path(concept["class_data_dir_1"])
            if not class_images_dir_1.exists():
                class_images_dir_1.mkdir(parents=True, exist_ok=True)
            if args.real_prior:
                assert (
                    class_images_dir_1 / "images"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt_1']}\" --class_data_dir {class_images_dir_1} --num_class_images {args.num_class_images}"
                assert (
                    len(list((class_images_dir_1 / "images").iterdir())) == args.num_class_images
                ), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt_1']}\" --class_data_dir {class_images_dir_1} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir_1 / "caption.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt_1']}\" --class_data_dir {class_images_dir_1} --num_class_images {args.num_class_images}"
                assert (
                    class_images_dir_1 / "images.txt"
                ).exists(), f"Please run: python retrieve.py --class_prompt \"{concept['class_prompt_1']}\" --class_data_dir {class_images_dir_1} --num_class_images {args.num_class_images}"
                concept["class_prompt_1"] = os.path.join(class_images_dir_1, "caption.txt")
                concept["class_data_dir_1"] = os.path.join(class_images_dir_1, "images.txt")
                args.concepts_list[i] = concept
                accelerator.wait_for_everyone()
            else:
                cur_class_images_1 = len(list(class_images_dir_1.iterdir()))

                if cur_class_images_1 < args.num_class_images:
                    torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                    if args.prior_generation_precision == "fp32":
                        torch_dtype = torch.float32
                    elif args.prior_generation_precision == "fp16":
                        torch_dtype = torch.float16
                    elif args.prior_generation_precision == "bf16":
                        torch_dtype = torch.bfloat16
                    pipeline = DiffusionPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        revision=args.revision,
                    )
                    pipeline.set_progress_bar_config(disable=True)

                    num_new_images = args.num_class_images - cur_class_images_1
                    logger.info(f"Number of class images to sample: {num_new_images}.")

                    sample_dataset = PromptDataset(args.class_prompt_1, num_new_images)
                    sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                    sample_dataloader = accelerator.prepare(sample_dataloader)
                    pipeline.to(accelerator.device)

                    for example in tqdm(
                        sample_dataloader,
                        desc="Generating class images",
                        disable=not accelerator.is_local_main_process,
                    ):
                        images = pipeline(example["prompt"]).images

                        for i, image in enumerate(images):
                            hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                            image_filename = (
                                class_images_dir_1 / f"{example['index'][i] + cur_class_images_1}-{hash_image}.jpg"
                            )
                            image.save(image_filename)

                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split("+")
        args.initializer_token = args.initializer_token.split("+")
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(
            args.modifier_token, args.initializer_token[: len(args.modifier_token)]
        ):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for x, y in zip(modifier_token_id, initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)
    ########################################################
    ########################################################

    vae.requires_grad_(False)
    if args.modifier_token is None:
        text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16" and args.modifier_token is not None:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    attention_class = (
        CustomDiffusionAttnProcessor
    )
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            attention_class = CustomDiffusionXFormersAttnProcessor
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # now we will add new Custom Diffusion weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers (2x attention layers -> 1x cross attention layer, 1x self attention layer)

    # Only train key, value projection layers if freeze_model = 'crossattn_kv' else train all params in the cross attention layer
    train_k = args.train_k
    train_v = args.train_v
    train_q = args.train_q
    train_out = args.train_out
    custom_diffusion_attn_procs = {}

    controller = AttentionStore(LOW_RESOURCE=False)
    controller.num_att_layers = 32

    st = unet.state_dict()
    for name, _ in unet.attn_processors.items():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        layer_name = name.split(".processor")[0]


        weights = {}
        if train_k:
            weights["to_k_custom_diffusion.weight"] = st[layer_name + ".to_k.weight"]
        if train_v:
            weights["to_v_custom_diffusion.weight"] = st[layer_name + ".to_v.weight"]
        if train_q:
            weights["to_q_custom_diffusion.weight"] = st[layer_name + ".to_q.weight"]
        if train_out:
            weights["to_out_custom_diffusion.0.weight"] = st[layer_name + ".to_out.0.weight"]
            weights["to_out_custom_diffusion.0.bias"] = st[layer_name + ".to_out.0.bias"]
        if cross_attention_dim is not None:
            custom_diffusion_attn_procs[name] = attention_class(
                train_k=train_k,
                train_v=train_v,
                train_q=train_q,
                train_out=train_out,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            ).to(unet.device)
            custom_diffusion_attn_procs[name].load_state_dict(weights)
        else:
            custom_diffusion_attn_procs[name] = attention_class(
                train_k=False,
                train_v=False,
                train_q=False,
                train_out=False,
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                controller=controller,
                place_in_unet=name.split("_")[0],
            )
    del st
    unet.set_attn_processor(custom_diffusion_attn_procs)
    custom_diffusion_layers = AttnProcsLayers(unet.attn_processors)

    accelerator.register_for_checkpointing(custom_diffusion_layers)


    #############################################################################
    #register CustomModelLoader as model saving hook
    loader = CustomModelLoader(unet=unet)
    #############################################################################

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate = args.learning_rate * 2.0

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    optimizer = optimizer_class(
        itertools.chain(text_encoder.get_input_embeddings().parameters(), custom_diffusion_layers.parameters())
        if args.modifier_token is not None
        else custom_diffusion_layers.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )


    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
        )
        .latent_dist.sample()
        .size()[-1],
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
        sample_mode = "all",
        object_list = args.object_list,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False, #True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )


    # Prepare everything with our `accelerator`.
    if args.modifier_token is not None:
        custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            custom_diffusion_layers, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    args.num_warmup_epochs = math.ceil(args.warmup_steps / num_update_steps_per_epoch)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    warmup_step = 0

    from_where = ["up", "down", "mid"]
    res_list = [16]
    self_res_list = [32]

    calculate_cross_attn_loss = False
    mask_token = None

    def jensen_shannon_divergence(p, q, eps=1e-16):

        # Softmax to make them probability distributions
        p_probs = F.softmax(p.view(-1), dim=0).view(p.size())
        q_probs = F.softmax(q.view(-1), dim=0).view(q.size())

        # Calculate the average of the probabilities
        m = 0.5 * (p_probs + q_probs)
        
        # Compute the Jensen-Shannon Divergence
        js_divergence = 0.5 * (torch.sum(p_probs * torch.log(p_probs + eps)) + torch.sum(q_probs * torch.log(q_probs + eps)))
        js_divergence -= torch.sum(m * torch.log(m + eps))
    
        return js_divergence
    
    def assign_classes_3_class(a, b, c, threshold):
        assert a.shape == b.shape == c.shape, 'All images must have the same shape'
        a = a.cpu().detach()
        b = b.cpu().detach()
        c = c.cpu().detach()
        classes = np.ones_like(a, dtype=np.uint8) * 3
        
        for count in range(3):
            all_maps = [a, b, c]
            map = all_maps.pop(count)
            diff_1 = map - all_maps[0]
            diff_2 = map - all_maps[1]

            classes[(diff_1 > threshold) & (diff_2 > threshold)] = count
        

        return classes
    
    def calculate_ssim(array1, array2):
        # Calculate SSIM
        array1 = array1.squeeze()
        array2 = array2.squeeze()

        ssim_value = ssim(array1,
                        array2,
                        multichannel=True,
                        win_size=7,
                        data_range=1)  # Set win_size explicitly

        return ssim_value
    
    def get_mask(self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced):
        # post-process the cross-attention map to get the mask for each subject, and then assign the attention cluster to its corresponding subject
        attention_segmentation = assign_classes_3_class(a = self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"][0] if 0 not in cross_attention_map_to_be_replaced else self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map_cross_power'][0], # now we only consider two subjects
                                                b = self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"][1] if 1 not in cross_attention_map_to_be_replaced else self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map_cross_power'][1],
                                                c = self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"][2], # the background, we don't consider the background for now
                                                threshold = args.attention_threshold
                                                )

        cross_attention_map_segmentation = {"Class_A": [], "Class_B": [], "Background": []}
        mask_token = {}
        for class_index, class_name in enumerate(cross_attention_map_segmentation.keys()):
            mask = np.zeros_like(attention_segmentation, dtype=np.float32)
            mask[attention_segmentation == class_index] = 1.0
            cross_attention_map_segmentation[class_name].append(mask)

            if class_index < len(cross_attention_map_segmentation):
                # calculate the element-wise multiplication of the cross attention map and the mask
                masked_cross_attention_map = []
                for scamp_id, scamcp in enumerate(self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"]):
                    if scamp_id in cross_attention_map_to_be_replaced:
                        scamcp = self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map_cross_power'][scamp_id]
                    masked_cross_attention_map.append(torch.sum(scamcp.cpu().detach() * mask))
                masked_cross_attention_map = torch.stack(masked_cross_attention_map)
                max_index = torch.argmax(masked_cross_attention_map)
                print(f'The mask {class_name} is assigned to identifier token {max_index}')
                mask_token[class_name] = max_index

        return mask_token, cross_attention_map_segmentation
    
    def get_attention_maps(prompts, bsz, calculate_cross_attn_loss, mask_token, tokens_id_to_use):
        
        # add attention loss to loss
        cross_attention_maps_1 = []
        cross_attention_maps_res_1 = []
        self_attention_maps_1 = []
        self_attention_maps_res_1 = []
        affinity_mat_1 = []
        affinity_mat_res_1 = []
        cross_attention_maps_1_power = []
        cross_attention_maps_res_1_power = []

        descriptive_maps = []
        descriptive_maps_res = []
        descriptive_maps_power = []
        descriptive_maps_power_res = []


        assert len(tokens_id_to_use) ==  bsz // 2 == 1, "now we only consider the batch size of 2 (with prior)"


        for i in range(bsz // 2):
            for res in res_list:
                cross_attention_map = aggregate_current_attention(prompts=prompts,
                                                                attention_store=controller, 
                                                                res=res, 
                                                                from_where=from_where,
                                                                is_cross=True,
                                                                select=i)

                target_indices, modifier_token_sets = find_the_identify_token_index(prompts[i], modifier_token_id)
                
                assert len(target_indices) == len(tokens_id_to_use[i]) + 1

                image_1 = cross_attention_map[:, :, target_indices]
                image_1 = F.interpolate(image_1.unsqueeze(0).permute(0, 3, 1, 2), (32, 32), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).squeeze(0)

                image_1_2 = []
                cross_attention_map_to_be_replaced = []
                for set_index, modifier_token_set in enumerate(modifier_token_sets):
                    if len(modifier_token_set) == 1:
                        image_1_2.append(torch.zeros_like(cross_attention_map[..., 0]))
                    elif len(modifier_token_set) > 1:
                        max_tensor, _ = torch.max(cross_attention_map[:, :, modifier_token_set], dim=-1)
                        image_1_2.append(max_tensor)
                        cross_attention_map_to_be_replaced.append(set_index)
                image_1_2 = torch.stack(image_1_2).permute(1, 2, 0)
                image_1_2 = F.interpolate(image_1_2.unsqueeze(0).permute(0, 3, 1, 2), (32, 32), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).squeeze(0)

                assert image_1_2.shape == image_1.shape, 'The shape of descriptive_maps and image_1 should be the same'


                image_1_normalized = torch.zeros_like(image_1)
                image_1_2_normalized = torch.zeros_like(image_1_2)
                for tar_ind in range(image_1.shape[2]):
                    image_1_normalized[..., tar_ind] = (image_1[..., tar_ind] - torch.min(image_1[..., tar_ind])) / (torch.max(image_1[..., tar_ind]) - torch.min(image_1[..., tar_ind]))
                    image_1_2_normalized[..., tar_ind] = (image_1_2[..., tar_ind] - torch.min(image_1_2[..., tar_ind])) / (torch.max(image_1_2[..., tar_ind]) - torch.min(image_1_2[..., tar_ind]))

                image_1_power = torch.pow(image_1, 2)     
                image_1_2_power = torch.pow(image_1_2, 2)

                image_1_power_normalized = torch.zeros_like(image_1_power)
                image_1_2_power_normalized = torch.zeros_like(image_1_2_power)
                for tar_ind in range(image_1_power.shape[2]):
                    image_1_power_normalized[..., tar_ind] = (image_1_power[..., tar_ind] - torch.min(image_1_power[..., tar_ind])) / (torch.max(image_1_power[..., tar_ind]) - torch.min(image_1_power[..., tar_ind]))
                    image_1_2_power_normalized[..., tar_ind] = (image_1_2_power[..., tar_ind] - torch.min(image_1_2_power[..., tar_ind])) / (torch.max(image_1_2_power[..., tar_ind]) - torch.min(image_1_2_power[..., tar_ind]))

                cross_attention_maps_res_1.append(image_1_normalized)
                descriptive_maps_res.append(image_1_2_normalized)

                cross_attention_maps_res_1_power.append(image_1_power_normalized)
                descriptive_maps_power_res.append(image_1_2_power_normalized)


            cross_attention_maps_1.append(torch.stack(cross_attention_maps_res_1).mean(dim=0))
            descriptive_maps.append(torch.stack(descriptive_maps_res).mean(dim=0))
            cross_attention_maps_1_power.append(torch.stack(cross_attention_maps_res_1_power).mean(dim=0))
            descriptive_maps_power.append(torch.stack(descriptive_maps_power_res).mean(dim=0))

            cross_attention_maps_res_1 = []
            descriptive_maps_res = []
            cross_attention_maps_res_1_power = []
            descriptive_maps_power_res = []


            for self_res in self_res_list:
                self_attention_map = aggregate_current_attention(prompts=prompts, #get the averaged self attention map from different layers (in same size) of the ith text in batch
                                    attention_store=controller, 
                                    res=self_res, 
                                    from_where=from_where,
                                    is_cross=False,
                                    select=i)
                
                self_attention_map = F.interpolate(self_attention_map.unsqueeze(0).permute(0, 3, 1, 2), (32, 32), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).squeeze(0)
                self_attention_map = (self_attention_map - torch.min(self_attention_map)) / (torch.max(self_attention_map) - torch.min(self_attention_map))
                affinity_mat = self_attention_map.reshape(32**2, 32**2)

                affinity_mat_multi = torch.matrix_power(affinity_mat, 4) 
                affinity_mat_multi = (affinity_mat_multi - torch.min(affinity_mat_multi)) / (torch.max(affinity_mat_multi) - torch.min(affinity_mat_multi))

                self_attention_maps_res_1.append(affinity_mat)
                affinity_mat_res_1.append(affinity_mat_multi)

            self_attention_maps_1.append(torch.stack(self_attention_maps_res_1).mean(dim=0))
            affinity_mat_1.append(torch.stack(affinity_mat_res_1).mean(dim=0))

            self_attention_maps_res_1 = []
            affinity_mat_res_1 = []



        cross_attention_maps_1 = torch.stack(cross_attention_maps_1).to("cuda")
        descriptive_maps = torch.stack(descriptive_maps).to("cuda")
        cross_attention_maps_1_power = torch.stack(cross_attention_maps_1_power).to("cuda")
        descriptive_maps_power = torch.stack(descriptive_maps_power).to("cuda")

        self_attention_maps_1 = torch.stack(self_attention_maps_1).to("cuda")
        affinity_mat_1 = torch.stack(affinity_mat_1).to("cuda")

        attention_loss = 0.0
        attention_loss_list = []
        if len(target_indices) == 1:
            self_cross_attention_map = (affinity_mat_1 @ cross_attention_maps_1.reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
            self_cross_attention_map = (self_cross_attention_map - torch.min(self_cross_attention_map)) / (torch.max(self_cross_attention_map) - torch.min(self_cross_attention_map))
            self_cross_attention_map_cross_power = (affinity_mat_1 @ cross_attention_maps_1_power.reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
            self_cross_attention_map_cross_power = (self_cross_attention_map_cross_power - torch.min(self_cross_attention_map_cross_power)) / (torch.max(self_cross_attention_map_cross_power) - torch.min(self_cross_attention_map_cross_power))


        else:
            self_cross_attention_map_for_all_subjects = {"self_attention_maps": self_attention_maps_1, "cross_attention_maps": cross_attention_maps_1, "cross_attention_maps_power": cross_attention_maps_1_power, 
                                                         "descriptive_maps": descriptive_maps, "descriptive_maps_power": descriptive_maps_power,
                                                        "self_cross_attention_map": [], "self_cross_attention_map_self_power": [], "self_cross_attention_map_cross_power": [], "self_cross_attention_map_cross_power_self_power": [],
                                                        'descriptive_self_cross_attention_map': [], 'descriptive_self_cross_attention_map_self_power': [], 'descriptive_self_cross_attention_map_cross_power': [], 'descriptive_self_cross_attention_map_cross_power_self_power': []}
            tokens_id_to_use[0].append(2)
            for tar_ind, tar_obj_ind in enumerate(tokens_id_to_use[0]):
                self_cross_attention_map = (affinity_mat_1 @ cross_attention_maps_1[..., tar_ind].reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
                self_cross_attention_map = (self_cross_attention_map - torch.min(self_cross_attention_map)) / (torch.max(self_cross_attention_map) - torch.min(self_cross_attention_map))
                self_cross_attention_map_for_all_subjects["self_cross_attention_map"].append(self_cross_attention_map)

                self_cross_attention_map_cross_power = (affinity_mat_1 @ cross_attention_maps_1_power[..., tar_ind].reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
                self_cross_attention_map_cross_power = (self_cross_attention_map_cross_power - torch.min(self_cross_attention_map_cross_power)) / (torch.max(self_cross_attention_map_cross_power) - torch.min(self_cross_attention_map_cross_power))
                self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"].append(self_cross_attention_map_cross_power)

                descriptive_self_cross_attention_map = (affinity_mat_1 @ descriptive_maps[..., tar_ind].reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
                descriptive_self_cross_attention_map = (descriptive_self_cross_attention_map - torch.min(descriptive_self_cross_attention_map)) / (torch.max(descriptive_self_cross_attention_map) - torch.min(descriptive_self_cross_attention_map))
                self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map'].append(descriptive_self_cross_attention_map)

                descriptive_self_cross_attention_map_cross_power = (affinity_mat_1 @ descriptive_maps_power[..., tar_ind].reshape(-1, 32**2, 1)).reshape(-1, 32, 32)
                descriptive_self_cross_attention_map_cross_power = (descriptive_self_cross_attention_map_cross_power - torch.min(descriptive_self_cross_attention_map_cross_power)) / (torch.max(descriptive_self_cross_attention_map_cross_power) - torch.min(descriptive_self_cross_attention_map_cross_power))
                self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map_cross_power'].append(descriptive_self_cross_attention_map_cross_power)


                if calculate_cross_attn_loss:
                    do_calculate = False
                    for mask_class, token_index in mask_token.items():
                        if tar_obj_ind == token_index:
                            ground_truth_mask  = torch.from_numpy(cross_attention_map_segmentation[mask_class][0]).to("cuda")
                            attn_loss = jensen_shannon_divergence(cross_attention_maps_1[..., tar_ind], ground_truth_mask)
                            do_calculate = True
                            break
                    if not do_calculate:
                        raise ValueError("The token index is not in the mask_token list")

            
                else:
                    attn_loss = torch.tensor(0.0).to("cuda")

                attention_loss_list.append(attn_loss)

        attention_loss_1 = torch.mean(torch.stack(attention_loss_list), dim=0)
        attention_loss = attention_loss_1


        return target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced, attention_loss
    
    def show_attention_maps(target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_segmentation):

        show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["self_attention_maps"], obj="self_attention_map")


        if len(target_indices) == 1:
            show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["self_cross_attention_map"], obj="self_cross_attention_map")
            show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"], obj="self_cross_attention_map_cross_power")
        else:
            for tar_ind in range(len(target_indices)):
                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["cross_attention_maps"][..., tar_ind], obj=f"cross_attention_{tar_ind}")
                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["cross_attention_maps_power"][..., tar_ind], obj=f"cross_attention_power_{tar_ind}")

                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["self_cross_attention_map"][tar_ind], obj=f"self_cross_attention_map_{tar_ind}")
                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["self_cross_attention_map_cross_power"][tar_ind], obj=f"self_cross_attention_map_cross_power_{tar_ind}")

                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["descriptive_maps"][..., tar_ind], obj=f"descriptive_cross_attention_{tar_ind}")
                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects["descriptive_maps_power"][..., tar_ind], obj=f"descriptive_cross_attention_power_{tar_ind}")

                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map'][tar_ind], obj=f"descriptive_self_cross_attention_map_{tar_ind}")
                show_attention_map_during_training(cross_attention_map = self_cross_attention_map_for_all_subjects['descriptive_self_cross_attention_map_cross_power'][tar_ind], obj=f"descriptive_self_cross_attention_map_cross_power_{tar_ind}")

            show_attention_map_during_training(cross_attention_map = torch.from_numpy(cross_attention_map_segmentation["Class_A"][0]), obj="cross_attention_map_segmentation_Class_A")
            show_attention_map_during_training(cross_attention_map = torch.from_numpy(cross_attention_map_segmentation["Class_B"][0]), obj="cross_attention_map_segmentation_Class_B")
            show_attention_map_during_training(cross_attention_map = torch.from_numpy(cross_attention_map_segmentation["Background"][0]), obj="cross_attention_map_segmentation_Background")


    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.warmup_steps-1),
        initial=initial_global_step,
        desc="Warmup Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # we first exucute the warmup steps to get the initial attention maps
    for epoch in range(first_epoch, args.num_warmup_epochs):
        unet.train()
        break_all = False
        if args.modifier_token is not None:
            text_encoder.train()  
        for warmup_step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):

                tokens_id_to_use = batch["token_ids"]

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, 300, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0]

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                if args.with_prior_preservation:
                    target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced, attention_loss = get_attention_maps(prompts = batch["input_ids"], 
                                                                                                                       bsz = bsz, 
                                                                                                                       calculate_cross_attn_loss = calculate_cross_attn_loss, 
                                                                                                                       mask_token = mask_token,
                                                                                                                       tokens_id_to_use = tokens_id_to_use)
                else:
                    raise ValueError("The with_prior_preservation should be True")

                if warmup_step >= args.warmup_steps - 1 or warmup_step == 0:

                    mask_token, cross_attention_map_segmentation = get_mask(self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced)

                    # for the mask initilization, we should make sure the identifier tokens for different classes are different
                    if mask_token["Class_A"] == mask_token["Class_B"] or mask_token["Class_A"] == mask_token["Background"] or mask_token["Class_B"] == mask_token["Background"]:
                        raise ValueError("The identifier tokens for Class_A and Class_B are the same, please check the threshold value or timestep for the mask initialization")
                    else:
                        mask_token_backup = mask_token
                        cross_attention_map_segmentation_backup = cross_attention_map_segmentation
                        calculate_cross_attn_loss = True


                    show_attention_maps(target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_segmentation)

                wandb.log({"denoise_loss": loss})
                if args.with_prior_preservation:
                    wandb.log({"prior_loss": prior_loss})
                wandb.log({"attention_loss": attention_loss})

                #clean controller after each sampling or training step
                controller.cur_step = 0
                controller.cur_att_layer = 0
                controller.attention_store = controller.get_empty_store()

                total_loss = (loss + attention_loss)
                wandb.log({"total_loss": total_loss})

                # do not optimize the model for the first 10 steps, this is the warm-up stage
                total_loss = total_loss * 0.0

                accelerator.backward(total_loss)

                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                        if args.modifier_token is not None
                        else custom_diffusion_layers.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                warmup_step += 1

            if warmup_step >= args.warmup_steps:
                break_all = True
                break
        if break_all:
            break



    del progress_bar
    del train_dataset
    del train_dataloader

    # Dataset and DataLoaders creation:
    train_dataset = CustomDiffusionDataset(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        mask_size=vae.encode(
            torch.randn(1, 3, args.resolution, args.resolution).to(dtype=weight_dtype).to(accelerator.device)
        )
        .latent_dist.sample()
        .size()[-1],
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        aug=not args.noaug,
        sample_mode = "subset",
        object_list = args.object_list,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_dataloader, lr_scheduler = accelerator.prepare(train_dataloader, lr_scheduler)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Training Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )


    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        if args.modifier_token is not None:
            text_encoder.train()  
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet), accelerator.accumulate(text_encoder):

                if global_step % 20 == 0:
                    tokens_id_to_use = [[0,1]* args.train_batch_size]
                    input_ids, prior_ids = torch.chunk(batch["input_ids"], 2, dim=0)
                    full_subject_input_ids = torch.cat([tokenizer(
                                    args.instance_prompt,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=tokenizer.model_max_length,
                                    return_tensors="pt",
                                ).input_ids] * args.train_batch_size, dim=0).to("cuda")
                    batch["input_ids"] = torch.cat([full_subject_input_ids, prior_ids], dim=0)

                else:
                    tokens_id_to_use = batch["token_ids"]

                instance_mask = {}
                for mask_class, token_index in mask_token.items():
                    assert token_index not in instance_mask, "The token index should not be repeated, please check the value of 'mask_token'"
                    instance_mask[token_index.item()] = cross_attention_map_segmentation[mask_class][0].squeeze()

                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor


                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                if train_dataloader.dataset.sample_mode == "subset":
                    batch_insatnce_mask_to_use = []
                    for ii in range(bsz // 2):
                        instance_mask_to_use = [torch.from_numpy(instance_mask[tokens_index]) for tokens_index in tokens_id_to_use[ii]]#[instance_mask[token_index.item()] for token_index in tokens_id_to_use[ii]]
                        instance_mask_to_use = torch.stack(instance_mask_to_use, dim=0)
                        batch_insatnce_mask_to_use.append(instance_mask_to_use)

                    batch_insatnce_mask_to_use = torch.stack(batch_insatnce_mask_to_use, dim=0).to("cuda")
                    batch_max_masks = torch.max(batch_insatnce_mask_to_use, dim=1).values
                    batch_max_masks = F.interpolate(batch_max_masks.unsqueeze(1), size=(64, 64), mode='bicubic').squeeze(1)
                    batch_max_masks = batch_max_masks.unsqueeze(1).repeat(1, 4, 1, 1)

                    # create the mask for the prior image
                    if args.with_prior_preservation:
                        batch_max_masks_prior = torch.ones_like(batch_max_masks)
                        batch_max_masks = torch.cat([batch_max_masks, batch_max_masks_prior], dim=0)
                else:
                    batch_max_masks = torch.ones_like(latents)


                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    mask = torch.chunk(batch["mask"], 2, dim=0)[0]

                    # apply the mask to the model_pred and target
                    model_pred = model_pred * batch_max_masks
                    target = target * batch_max_masks

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    mask = batch["mask"]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = ((loss * mask).sum([1, 2, 3]) / mask.sum([1, 2, 3])).mean()


                if args.with_prior_preservation:
                    target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced, attention_loss = get_attention_maps(prompts = batch["input_ids"], 
                                                                                                                       bsz = bsz, 
                                                                                                                       calculate_cross_attn_loss = calculate_cross_attn_loss, 
                                                                                                                       mask_token = mask_token,
                                                                                                                       tokens_id_to_use = tokens_id_to_use)
                else:
                    raise ValueError("The with_prior_preservation should be True")




                if global_step % 20 == 0:
                    mask_token, cross_attention_map_segmentation = get_mask(self_cross_attention_map_for_all_subjects, cross_attention_map_to_be_replaced)
                        
                    ssim_score_A = calculate_ssim(cross_attention_map_segmentation["Class_A"][0], cross_attention_map_segmentation_backup["Class_A"][0])
                    ssim_score_B = calculate_ssim(cross_attention_map_segmentation["Class_B"][0], cross_attention_map_segmentation_backup["Class_B"][0])

                    if mask_token["Class_A"] == mask_token["Class_B"] or mask_token["Class_A"] == mask_token["Background"] or mask_token["Class_B"] == mask_token["Background"] or ssim_score_A < args.ssim_threshold or ssim_score_B < args.ssim_threshold:
                        mask_token = mask_token_backup
                        cross_attention_map_segmentation = cross_attention_map_segmentation_backup
                    else:
                        mask_token_backup = mask_token
                        cross_attention_map_segmentation_backup = cross_attention_map_segmentation


                    show_attention_maps(target_indices, self_cross_attention_map_for_all_subjects, cross_attention_map_segmentation)


                wandb.log({"denoise_loss": loss})
                if args.with_prior_preservation:
                    wandb.log({"prior_loss": prior_loss})
                wandb.log({"attention_loss": attention_loss})

                #clean controller after each sampling or training step
                controller.cur_step = 0
                controller.cur_att_layer = 0
                controller.attention_store = controller.get_empty_store()


                total_loss = loss + attention_loss
                wandb.log({"total_loss": total_loss})


                accelerator.backward(total_loss)


                # Zero out the gradients for all token embeddings except the newly added
                # embeddings for the concept, as we only want to optimize the concept embeddings
                if args.modifier_token is not None:
                    if accelerator.num_processes > 1:
                        grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                    else:
                        grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                    # Get the index for tokens that we want to zero the grads for
                    index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                    for i in range(len(modifier_token_id[1:])):
                        index_grads_to_zero = index_grads_to_zero & (
                            torch.arange(len(tokenizer)) != modifier_token_id[i]
                        )
                    grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[
                        index_grads_to_zero, :
                    ].fill_(0)

                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(text_encoder.parameters(), custom_diffusion_layers.parameters())
                        if args.modifier_token is not None
                        else custom_diffusion_layers.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)


            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step > 99 and global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        unet = unet.to(torch.float32)
                        loader.save_attn_procs(save_path, 
                                               safe_serialization=not args.no_safe_serialization)
                        save_new_embed(
                            text_encoder,
                            modifier_token_id,
                            accelerator,
                            args,
                            save_path,
                            safe_serialization=not args.no_safe_serialization,
                        )


            if global_step >= args.max_train_steps:
                break



    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)