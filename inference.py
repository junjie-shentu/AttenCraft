from diffusers import  StableDiffusionPipeline
import torch
import os
from tqdm import tqdm

from utils.CustomModelLoader_for_inference import CustomModelLoader

from utils.CrossAttnMap import AttentionStore

controller = AttentionStore(LOW_RESOURCE=False)
controller.num_att_layers = 16

def build_pipeline(ckpt_path):

    # Load the pipeline with the same arguments (model, revision) that were used for training
    model_id = "stabilityai/stable-diffusion-2-1-base"

    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_safetensors=True).to("cuda")

    #register unet for loader
    if "wq" in ckpt_path:
        train_q = True
    else:
        train_q = False
    if "wk" in ckpt_path:
        train_k = True
    else:
        train_k = False
    if "wv" in ckpt_path:
        train_v = True
    else:
        train_v = False
    if "wout" in ckpt_path:
        train_out = True
    else:
        train_out = False
    
    loader = CustomModelLoader(pipeline.unet)
    loader.load_attn_procs(ckpt_path, weight_name="pytorch_custom_diffusion_weights.bin", train_q=train_q, train_k=train_k, train_v=train_v, train_out=train_out, controller=controller)

    pipeline.load_textual_inversion(ckpt_path, weight_name="<new1>.bin")
    pipeline.load_textual_inversion(ckpt_path, weight_name="<new2>.bin")

    
    return pipeline


def generate_image(ckpt_path, text_prompt_list):

    #get the pipeline
    pipeline = build_pipeline(ckpt_path)

    all_generated_images = {}
    for text_prompt in tqdm(text_prompt_list, desc='Text Prompt Loop'):
        all_generated_images[text_prompt] = []
        for seed in tqdm(RANDOM_SEED, desc='Seed Loop'):
            generator = torch.Generator("cuda").manual_seed(seed)
            images = pipeline(prompt=text_prompt, num_images_per_prompt=10, num_inference_steps=50, generator = generator).images
            all_generated_images[text_prompt].extend(images)

    return all_generated_images


def save_generated_images(all_generated_images, image_output_path):
    for text_prompt, image_list in all_generated_images.items():
        for i, image in enumerate(image_list):
            image.save(os.path.join(image_output_path, "image", f"{text_prompt}_{i+RANDOM_SEED[0]}.jpg"))



def generate_and_save_image(ckpt_path, text_prompt_list, image_output_path):
    
    all_generated_images = generate_image(ckpt_path, text_prompt_list)

    save_generated_images(all_generated_images, image_output_path)



if __name__ == "__main__":

    NEW_TOKEN_1 = "<new1>"
    NEW_TOKEN_2 = "<new2>"
    
    RANDOM_SEED = [42] # list of random seed, each seed will generate 10 images per text prompt


    text_prompt_list = ["A list of the text prompt"]


    ckpt_path_list = ["list of the paths containing the checkpoint files"]
                

    image_output_path_list = ["list of the paths to save the generated images"]
    
    for ckpt_path, image_output_path in zip(ckpt_path_list, image_output_path_list):
        os.makedirs(os.path.join(image_output_path, "image"), exist_ok=True)

        result = generate_and_save_image(ckpt_path, text_prompt_list, image_output_path)




    



