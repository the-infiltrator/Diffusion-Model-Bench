import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import os
import glob
from PIL import Image


def plot_metrics(metrics_file="logs/metrics.txt", plots_dir="plots"):
    """Plot the metrics"""

    # Load metrics from file
    df = pd.read_csv(metrics_file)

    # Average metrics for each model
    avg_metrics = df.groupby('model').mean().reset_index()

    # Normalize the metrics
    scaler = MinMaxScaler()
    avg_metrics_scaled = pd.DataFrame(scaler.fit_transform(
        avg_metrics.iloc[:, 1:]), columns=avg_metrics.columns[1:], index=avg_metrics['model'])

    # Compute a new metric that combines the three metrics
    avg_metrics_scaled['score'] = avg_metrics_scaled.mean(axis=1)

    # Set the style to ggplot
    plt.style.use('ggplot')

    # Create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # Plot Time
    sns.barplot(x='elapsed_time', y='model', data=avg_metrics,
                palette='CMRmap_r', ax=axs[0, 0], alpha=0.75)
    axs[0, 0].set_title('Average Time per Model')
    axs[0, 0].set_xlabel('Time (seconds)')
    axs[0, 0].set_ylabel('Model')

    # Plot GPU Memory
    sns.barplot(x='gpu_memory', y='model', data=avg_metrics,
                palette='CMRmap_r', ax=axs[0, 1], alpha=0.75)
    axs[0, 1].set_title('Average GPU Peak Utilisation per Model')
    axs[0, 1].set_xlabel('Peak GPU Memory Utilisation (GB)')
    axs[0, 1].set_ylabel('Model')

    # Plot RAM Memory
    sns.barplot(x='ram_memory', y='model', data=avg_metrics,
                palette='CMRmap_r', ax=axs[1, 0], alpha=0.75)
    axs[1, 0].set_title('Average RAM Memory per Model')
    axs[1, 0].set_xlabel('RAM Memory (GB)')
    axs[1, 0].set_ylabel('Model')

    # Plot Score
    sns.barplot(x='score', y=avg_metrics_scaled.index,
                data=avg_metrics_scaled, palette='CMRmap_r', ax=axs[1, 1], alpha=0.75)
    axs[1, 1].set_title(
        'Overall Performance Score per Model (Lower is better)')
    axs[1, 1].set_xlabel('Score')
    axs[1, 1].set_ylabel('Model')

    # Automatically adjust subplot params so that the subplot fits into the figure area
    plt.tight_layout()

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(plots_dir, 'metrics_plot.png'))

    # Display the figure
    plt.show()


def plot_generated_images(original_dir, generated_dir, plots_dir="plots"):
    """Plot original and generated images side by side."""
    original_images = sorted(os.listdir(original_dir))
    generated_images = sorted(os.listdir(generated_dir))

    assert len(original_images) == len(
        generated_images), "Number of original and generated images must match."

    fig, axs = plt.subplots(len(original_images), 2,
                            figsize=(10, len(original_images)*5))

    for original_img in original_images:
        original = Image.open(os.path.join(original_dir, original_img))

        # find the corresponding generated image
        matching = [s for s in generated_images if original_img in s]
        if len(matching) != 1:
            print(
                f"Couldn't find a unique matching generated image for {original_img}")
            continue
        generated_img = matching[0]

        generated = Image.open(os.path.join(generated_dir, generated_img))

        i = original_images.index(original_img)
        axs[i, 0].imshow(original)
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis('off')

        axs[i, 1].imshow(generated)
        axs[i, 1].set_title("Generated Image")
        axs[i, 1].axis('off')

    plt.tight_layout()

    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)

    # Save the plot
    plt.savefig(os.path.join(plots_dir, 'img2img_generated.png'))

    # Display the plot
    plt.show()
