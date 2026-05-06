from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def plot_analyzed_latents(latent_counts_sorted, indices, k=1, TOP_N_INDICES=20):
    print(f"Top {k} latent counts: {latent_counts_sorted[:TOP_N_INDICES]}")
    print(f"Top {k} latent indices: {indices[:TOP_N_INDICES]}")
    plt.figure(figsize=(10, 3))
    plt.title(f"Top {k} Latent Counts")
    plt.bar(np.arange(len(latent_counts_sorted)), latent_counts_sorted.cpu().numpy(), width=0.5)
    plt.xticks(np.arange(len(latent_counts_sorted)), indices.cpu().numpy(), rotation=90)
    plt.xlabel("Latent Index")
    plt.ylabel("Count")
    plt.xlim(0, TOP_N_INDICES)  # Limit x-axis to first 1000 latents
    plt.show()

def create_top_activating_image_grid(activations, sae_id, dataset, num_rows = 2, num_cols = 5, mode = "show", top_images=None, filter_threshold = None):
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    if top_images is None:
        top_images = activations[:, sae_id].topk(num_rows * num_cols)
        top_images_indices = top_images.indices
        top_images_values = top_images.values
    else:
        top_images_indices = top_images[0]
        top_images_values = top_images[1]

    # turn top_images_indices numpy arry and top_images_values into a array
    if isinstance(top_images_indices, torch.Tensor):
        top_images_indices = top_images_indices.cpu().numpy()
    if isinstance(top_images_values, torch.Tensor):
        top_images_values = top_images_values.cpu().numpy()

    # turn into lists
    top_images_indices = top_images_indices.tolist()
    top_images_values = top_images_values.tolist()

    if filter_threshold is not None:
        # Filter out images with activation below threshold
        top_images_indices = [idx for idx, val in zip(top_images_indices, top_images_values) if val >= filter_threshold]
        top_images_values = [val for val in top_images_values if val >= filter_threshold]
        if len(top_images_indices) < num_rows * num_cols:
            print(f"Warning: Not enough images with activation >= {filter_threshold}. Showing {len(top_images_indices)} images instead.")
    for i, axi in enumerate(ax.ravel()):
        if i >= len(top_images_indices):
            break
        axi.imshow(dataset[top_images_indices[i]][0])
        axi.axis('off')
        axi.set_title(f'Image {top_images_indices[i]} with value {top_images_values[i]:.2f}')
    plt.tight_layout()
    if mode == "show":
        plt.show()
    elif mode == "save":
        os.makedirs('./sae_images', exist_ok=True)
        plt.savefig(f'./sae_images/sae_{sae_id}.png')
    elif mode == "return":
        return fig


def create_image_for_labelling(sae_id, topk_indices, topk_values, dataset, output_path,
                                   grid_size: tuple = (2, 5), filter_threshold = None) -> Image.Image:
    """
    Create a grid of top activating images for a neuron.

    Args:
        sae_id: ID of the SAE neuron
        topk_indices: Indices of top activating images
        topk_values: Values of top activating images
        dataset: Dataset object for accessing images
        output_path: Path to save the generated image
        grid_size: (rows, cols) for the image grid
        filter_threshold: Optional threshold to filter images by activation value
    Returns:
        None
    """
    num_rows, num_cols = grid_size
    fig = create_top_activating_image_grid(
        activations=None,
        sae_id=sae_id,
        dataset=dataset,
        num_rows=num_rows,
        num_cols=num_cols,
        mode="return",
        filter_threshold=filter_threshold,
        top_images=(topk_indices, topk_values)
    )

    # Convert Matplotlib figure to PIL Image in a clean way
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_grid = Image.open(buf).convert("RGB")
    buf.close()

    plt.close(fig)
    img_grid.save(output_path)