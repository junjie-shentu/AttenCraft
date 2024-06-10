from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
import torch.nn.functional as nnf
import numpy as np
import abc
from PIL import Image
import math
import cv2

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if self.low_resource else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross, place_in_unet):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.low_resource:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                self.forward(attn, is_cross, place_in_unet)

        self.cur_att_layer = self.cur_att_layer + 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step = self.cur_step + 1
            self.between_steps()

    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, LOW_RESOURCE=False):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

        self.low_resource = LOW_RESOURCE
        self.training_cross_attention_layers = []


    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross, place_in_unet):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 64 ** 2:
            self.step_store[key].append(attn)



    def between_steps(self):
        self.cur_step_attention = self.get_empty_store()
        for key in self.step_store:
            for i in range(len(self.step_store[key])):
                clone = self.step_store[key][i].clone()
                self.cur_step_attention[key].append(clone)

        if len(self.attention_store["down_cross"]) == 0:
            for key in self.step_store:
                for i in range(len(self.step_store[key])):
                    clone = self.step_store[key][i].clone()
                    self.attention_store[key].append(clone)

        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    clone = self.step_store[key][i].clone()
                    self.attention_store[key][i] = self.attention_store[key][i] + clone

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention
    
    def get_current_attention(self):
        return self.cur_step_attention

    def get_training_attention_layers(self):
        return self.training_cross_attention_layers


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.cur_step_attention = self.get_empty_store()

    def __init__(self, LOW_RESOURCE=False):
        super(AttentionStore, self).__init__(LOW_RESOURCE = LOW_RESOURCE)
        self.step_store = self.get_empty_store()
        self.attention_store = self.get_empty_store()
        self.cur_step_attention = self.get_empty_store()


def aggregate_current_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    # prompts: batch["input_ids"] -> len(prompts) = batch_size
    attention_maps = attention_store.get_current_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                #self-attention:cross_maps -> [batch_size, num_heads, res, res, res**2][select] -> [num_heads, res, res, res**2]
                #cross-attention:cross_maps -> [batch_size, num_heads, res, res, 77][select] -> [num_heads, res, res, 77]
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out

def aggregate_attention(prompts, attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    cross_attention = view_images(np.stack(images, axis=0))
    return cross_attention


def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    return pil_img









def show_all_cross_attention_maps(tokenizer, prompts, attention_store: AttentionStore, res: int, from_where: List[str], num_prompts: int):
    """Applies to the case where the prompts are all the same."""
    tokens = tokenizer.encode(prompts)
    decoder = tokenizer.decode

    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_cross"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(num_prompts, -1, res, res, item.shape[-1])#[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=1)
    out = out.sum(1) / out.shape[1]

    all_cross_attention = []
    for j in range(num_prompts):
        images = []
        for i in range(len(tokens)):
            image = out.cpu()[j, :, :, i]#attention_maps[:, :, i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((256, 256)))
            image = text_under_image(image, decoder(int(tokens[i])))
            images.append(image)
        cross_attention = view_images(np.stack(images, axis=0))
        all_cross_attention.append(cross_attention)

    return all_cross_attention




def show_cross_attention_specific_token(token_pos, prompts, attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    image = attention_maps[:, :, token_pos]
    image = 255 * image / image.max()
    image = image.unsqueeze(-1).expand(*image.shape, 3)
    image = image.numpy().astype(np.uint8)
    image = np.array(Image.fromarray(image).resize((256, 256)))
    images.append(image)
    cross_attention = view_images(np.stack(images, axis=0))
    return cross_attention
