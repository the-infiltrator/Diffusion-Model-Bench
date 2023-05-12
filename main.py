# main.py
from setup import install_dependencies
# Check and install dependencies
print("Checking and installing dependencies...")
install_dependencies()


from plotting import plot_metrics, plot_generated_images
from model import model_bench
from utils import load_images_from_directory_parallel,  find_latest_file
import argparse
import yaml
import os
import random


def main(spec):

    if not spec['plot_only']:

        print("Running the model benchmark...")
        # Run the model benchmark
        metrics = model_bench(spec)

    print("Plotting the metrics...")
    # Use the provided metrics file if one was provided, else use the latest
    metrics_file = spec['metrics_file'] if spec['metrics_file'] else find_latest_file(
        os.path.join(os.getcwd(), "logs/metrics_*.txt"))
    plot_metrics(metrics_file)

    print("Plotting the generated images...")
    # Plot the generated images
    plot_generated_images(spec['image_dir'], spec['generated_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model benchmark.')
    parser.add_argument('-s', '--spec_file', type=str,
                        required=False, help='Path to the spec.yaml file.')
    parser.add_argument('-m', '--model_file', type=str,
                        required=False, help='Path to the model file.')
    parser.add_argument('-i', '--image_dir', type=str, required=False,
                        help='Path to the directory containing the images.')
    parser.add_argument('-g', '--generated_dir', type=str, required=False,
                        help='Path to the directory where the generated images should be saved.')
    parser.add_argument('-p', '--prompt', type=str, default="A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by Greg Rutkowski and Thomas Kinkade, Trending on Artstation",
                        help='Prompt for the image generation.')
    parser.add_argument('-d', '--device', type=str, default='cuda',
                        help='Device to run the model on. Default is "cuda".')
    parser.add_argument('-tf', '--tf32', type=bool, default=False,
                        help='Use the TensorFloat32 (TF32) mode for faster but slightly less accurate computations.')
    parser.add_argument('--plot_only', action='store_true',
                        help='Only plot the latest metrics and images, do not run the model.')
    args = parser.parse_args()

    # try to load the parameters from the YAML file if it is provided. If no YAML file is provided, look for the parameters in the command-line argument
    if args.spec_file:
        with open(args.spec_file, 'r') as stream:
            try:
                spec = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit(1)
    else:
        spec = {
            "model_file": args.model_file,
            "image_dir": args.image_dir,
            "generated_dir": args.generated_dir,
            "prompt": args.prompt,
            "device": args.device,
            "tf32": args.tf32,
            "plot_only": args.plot_only,
            "metrics_file": None,
            "model": {"strength": 1,
                      "guidance_scale": 7.5,
                      "num_inference_steps": 25
                      }

        }

    main(spec)
