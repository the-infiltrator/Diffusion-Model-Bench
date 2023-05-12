# 🎨 Diffusion Model Bench: Your Img2Img Diffusion Playground! 🖼️
<p align="center">
<img src="https://github.com/the-infiltrator/Diffusion-Model-Bench/blob/main/preview.png" width="600">
</p>

*The image above illustrates an example of input to output transformation using our Img2Img Bench.*

## 📚 Introduction

Welcome to **Diffusion Model Bench** (DMB for short), your one-stop solution to test, play, and benchmark different image-to-image transformation models. You are free to set parameters, scale from a handful of images to thousands, and let the magic happen! The aim is to make this as flexible and adaptable as possible. You can also plot metrics and generated images to visualize your experiment's results. 📈🖌️

## 🏗️ Project Structure

The DMB codebase is organized into these primary files, each with its specific functionalities:

1. `utils.py`: This file houses helper functions necessary for the operations of the project. 

    - `find_latest_file(pattern)`: This function is used to find the most recent file that matches a given pattern.
    - `load_and_scale_image(image_path, size=512)`: This function loads and scales an image from a given path.
    - `load_images_from_directory_parallel(image_dir, max_workers=4)`: This function loads and scales images from a given directory into a list in a parallel manner for efficiency.
    - `load_model_names(model_file)`: This function loads model names from a file.

2. `setup.py`: This file is responsible for setting up the environment for running the project. 

    - `install_dependencies()`: This function checks and installs all necessary dependencies to run the project.

3. `model.py`: This file contains the core functions for image transformation and benchmarking. 

    - `img2img(image_tuple, model, prompt, device, generated_dir, strength, guidance_scale, num_inference_steps)`: This function performs image-to-image transformation using a given model.
    - `model_bench(spec)`: This function runs the `img2img` function for each image and a randomly selected model.

4. `main.py`: This is the main script that coordinates the entire benchmarking process, parses command-line arguments, runs the benchmark, and plots the metrics and generated images. It begins by installing dependencies, then it runs the model benchmark if `plot_only` is not specified. Finally, it plots the metrics and the generated images.

    - `main(spec)`: This function is the main function for running the benchmark. It takes a specification dictionary as an argument, which contains all the necessary parameters for running the benchmark.
    
Each of these files contributes to a modular and maintainable codebase, making it easy to modify and scale the project according to your needs.


## 🚀 Getting Started


Running the Diffusion Model Bench is super straightforward! Just use the `main.py` script. You can provide your parameters either via command-line arguments or a YAML file. Here's the lowdown on what each of these command-line parameters means:

- `-s` or `--spec_file`: Path to your YAML spec file. This is optional if you provide all other parameters via the command-line.
- `-m` or `--model_file`: Path to the .txt file containing your model names.
- `-i` or `--image_dir`: Path to the directory containing your input images.
- `-g` or `--generated_dir`: Path to the directory where you want to save your generated images.
- `-p` or `--prompt`: The prompt for image generation. By default, it's a beautiful painting description.
- `-d` or `--device`: The device to run the model on. It's 'cuda' by default, but could be 'cpu' if that's your thing.
- `-tf` or `--tf32`: If true, enables TensorFloat32 (TF32) mode for faster, albeit slightly less accurate computations.
- `--plot_only`: If this flag is provided, it will only plot the latest metrics and images without running the model.

To run the benchmark, simply use the `main.py` script. This script accepts command-line arguments for a variety of parameters. You can run the following with flexibility on the command-line-parameters mentioned above:

```shell
python main.py -m /path/to/model_names.txt -i /path/to/sourceimages -g /path to/generatedimages 
```




Alternatively, for better convinience you can also provide a YAML specification file for these parameters. 
```shell
python main.py -s spec.yaml 
```

In the command above, `-s` specifies the YAML file. The YAML file structure allows you to specify all the parameters in one place. Here's a brief look at the structure:

```yaml
model_file: '/path/to/model_names.txt'
image_dir: '/path/to/sourceimages'
generated_dir: /path/to/generatedimages'
prompt: 'A beautiful painting of a singular lighthouse...'
device: 'cuda'
tf32: false
metrics_file: null
plot_only: false
model:
  strength: 1
  guidance_scale: 7.5
  num_inference_steps: 25
```

### Requirements 🛠️

Before you can start transforming images like a pro, make sure you have these packages installed which is super easy through the `install_dependecies` function in `setup.py` which will automatically check and install all dependencies everytime you run the code:

- diffusers
- transformers
- scipy
- ftfy
- accelerate
- torchvision
- triton
- matplotlib
- huggingface_hub
- tqdm
- seaborn
- pandas
- scikit-learn
- Pillow
- pyyaml
## 📊 Output & Results


Below you can find the visualization of the performance metrics obtained by running the bench.

<p align="center">
<img src="https://github.com/the-infiltrator/Diffusion-Model-Bench/blob/main/plots/metrics_plot.png?raw=true" width="1000">
</p>

*The image above showcases the performance metrics outputted by our Img2Img Bench for various diffusion models.*

Your experiment results are neatly organized into directories:

- `logs`: This contains a log file recording the metrics (elapsed time, GPU memory, and RAM memory) for each image and model combination. Each log file is timestamped, making it easy to keep track of different runs.

- `plots`: This is where the visual magic happens. It contains plots of the metrics, giving you a bird's eye view of the benchmark results.

## 🏎️ Speed Improvements

The Diffusion Model Bench (DMB) is designed to be fast, efficient, and scalable. Here are some of the strategies we've employed to speed up the img2img transformation:

1. **Rescaling Images**: Images are rescaled to 512 pixels along the shortest edge while maintaining aspect ratio. This is in line with stable diffusion input size guidelines.

2. **Parallel Image Loading**: This capability allows for scaling up to a large number of images.

3. **DPM-Solver Usage**: We use DPM-Solver (and the improved version DPM-Solver++) for faster, high-quality sampling.

4. **TensorFloat32 (TF32) Usage**: On Ampere and later CUDA devices, you can use TensorFloat32 (TF32) through the `--tf32` flag for faster computations with typically negligible loss of numerical accuracy.

5. **Half Precision Weights**: For more GPU memory savings and speed, we load and run the model weights directly in half precision, specifically fp16.

6. **Sliced Attention**: This strategy allows for even additional memory savings. It performs the computation in steps instead of all at once.

7. **CPU Offloading**: We offload the weights to CPU and only load them to GPU when performing the forward pass. This results in significant memory savings.

8. **Chained Offloading with Attention Slicing**: By using the above technique in conjunction, we achieve minimal GPU memory consumption.

9. **Memory Management**: To further optimize memory utilization, `gc.collect()` and `torch.cuda.empty_cache()` are used after every inference run. This helps free up unused memory and ensures that our GPU utilization remains low.

These performance improvements are all about ensuring that you can transform as many images as quickly as possible without overloading your system's resources.


### Happy Diffusing! 🎉






