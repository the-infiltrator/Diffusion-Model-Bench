import os
import time
import random
import torch
import psutil
import gc
import datetime
from torch import autocast
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler
from utils import load_model_names, load_images_from_directory_parallel


def img2img(image_tuple, model, prompt, device, generated_dir, strength, guidance_scale, num_inference_steps):
    """Perform image-to-image transformation using a given model."""
    # unpack the image tuple
    image_name, init_img = image_tuple
    print(f"Processing image: {image_name} with model: {model}")

    # Record the initial memory usage
    initial_gpu_memory = torch.cuda.memory_allocated()
    initial_ram_memory = psutil.virtual_memory().used

    print("Initializing the pipeline...")
    # Initialize the pipeline
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16 if torch.cuda.is_available(
        ) and model != "stabilityai/stable-diffusion-2-1" else torch.float32,
        safety_checker=None,
        feature_extractor=None,
        use_auth_token=False
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config)
    pipe = pipe.to(device)
    print("Pipeline initialized.")

    # sequential cpu offloading for
    pipe.enable_sequential_cpu_offload()

    # attention slicing to save vram
    pipe.enable_attention_slicing(1)

    # Record the start time
    start_time = time.time()

    print("Performing image generation...")
    # Perform the image generation
    generator = torch.Generator(device=device).manual_seed(1024)

    if model != "stabilityai/stable-diffusion-2-1":
        with autocast(device):
            image = pipe(prompt=prompt, image=init_img, strength=strength,
                         guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images[0]
    else:
        image = pipe(prompt=prompt, image=init_img, strength=strength,
                     guidance_scale=guidance_scale, num_inference_steps=num_inference_steps, generator=generator).images[0]

    # Record the end time
    end_time = time.time()

    # Record the final memory usage
    final_gpu_memory = torch.cuda.memory_allocated()
    final_ram_memory = psutil.virtual_memory().used

    # Save the image
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    image_path = os.path.join(generated_dir, f'generated_{image_name}.png')
    image.save(image_path)
    print(f"Image generation completed. Generated image saved at {image_path}")

    # Clean up
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate the metrics
    elapsed_time = end_time - start_time
    gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # convert bytes to GB
    ram_memory = (final_ram_memory - initial_ram_memory) / \
        (1024 ** 3)  # convert bytes to GB
    print(
        f"Elapsed time: {elapsed_time}, GPU memory: {gpu_memory}, RAM memory: {ram_memory}")

    return image_name, model, elapsed_time, gpu_memory, ram_memory


def model_bench(spec):
    """Run the img2img function for each image and model."""

    # Use the TensorFloat32 (TF32) mode for faster but slightly less accurate computations
    torch.backends.cuda.matmul.allow_tf32 = spec['tf32']
    models = load_model_names(spec['model_file'])
    metrics = []

    # Get the current date and time
    now = datetime.datetime.now()

    # Format the datetime object as a string
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create a logs directory if it does not exist
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Append the timestamp to the filename and place it in the logs directory
    metrics_file = os.path.join(logs_dir, f"metrics_{timestamp}.txt")

    # Create metrics file if it doesn't exist
    if not os.path.exists(metrics_file):
        with open(metrics_file, 'w') as f:
            f.write("image_name,model,elapsed_time,gpu_memory,ram_memory\n")

    print("Loading images...")
    # Load images in parallel
    image_tuples = load_images_from_directory_parallel(spec['image_dir'])

    for image_tuple in image_tuples:
        # select a random model for each image
        model = random.choice(models)
        image_name, model, elapsed_time, gpu_memory, ram_memory = img2img(
            image_tuple, model, spec['prompt'], spec['device'], spec['generated_dir'], spec['model']['strength'], spec['model']['guidance_scale'], spec['model']['num_inference_steps'])

        metric = {
            'image_name': image_name,
            'model': model,
            'elapsed_time': elapsed_time,
            'gpu_memory': gpu_memory,
            'ram_memory': ram_memory
        }
        metrics.append(metric)

        # write the metric to file
        with open(metrics_file, 'a') as f:
            f.write(
                f"{image_name},{model},{elapsed_time},{gpu_memory},{ram_memory}\n")

    return metrics
