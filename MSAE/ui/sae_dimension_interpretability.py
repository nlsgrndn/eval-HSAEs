import gradio as gr
import pandas as pd
import os
from PIL import Image
from my_utils import load_dataset
from utils_figure_generation import create_top_activating_image_grid
from activations_preprocessing.utils_sae_activations import load_precomputed_top_k, get_precomputed_top_k_path
from my_config import DEFAULT_CONFIG
from path_hub import PathBuilder
import matplotlib.pyplot as plt
_PB = PathBuilder()

SAE_DIMENSIONS_LIST = list(range(DEFAULT_CONFIG.sae_model.width))
TOP_ACTIVATING_IMAGES_FOLDER_PATH = os.path.join(_PB.get_sae_images_path())
INTERPRETABILITY_CSV = _PB.get_interpretability_data_path()

def load_interpretability_data():
    """Load the interpretability CSV data."""
    try:
        return pd.read_csv(INTERPRETABILITY_CSV)
    except FileNotFoundError:
        return pd.DataFrame()

def get_top_activating_image_grid(sae_dimension):
    """Get the single image grid showing the top activating images"""
    file_name = f"neuron_{sae_dimension}_top_activating_images.png"
    dimension_file = os.path.join(TOP_ACTIVATING_IMAGES_FOLDER_PATH, file_name)

    if not os.path.exists(dimension_file):
        return None

    try:
        img = Image.open(dimension_file)
        return img
    except Exception as e:
        print(f"Error loading image {dimension_file}: {e}")
        return None
    
def get_top_activating_image(sae_dimension):
    precomputed_path = get_precomputed_top_k_path()
    top_k_activating_images_per_sae_id, _, _ = load_precomputed_top_k(precomputed_path)
    top_activating_image_per_sae_id = top_k_activating_images_per_sae_id[:, 0]
    dataset = load_dataset(DEFAULT_CONFIG.graph_eval_dataset, return_file_path=True)
    dataset_id = top_activating_image_per_sae_id[sae_dimension]
    top_image_path = dataset[dataset_id][0]
    if not os.path.exists(top_image_path):
        return None

    try:
        img = Image.open(top_image_path)
        return img
    except Exception as e:
        print(f"Error loading image {top_image_path}: {e}")
        return None

def get_top_activating_image_grid_on_the_fly(sae_dimension):
    TOPK = 10
    precomputed_path = get_precomputed_top_k_path()
    top_k_activating_images_per_sae_id, top_k_values_per_sae_id, _ = load_precomputed_top_k(precomputed_path)
    top_activating_images = top_k_activating_images_per_sae_id[sae_dimension, :TOPK]
    top_activating_values = top_k_values_per_sae_id[sae_dimension, :TOPK]
    dataset = load_dataset(DEFAULT_CONFIG.graph_eval_dataset, return_file_path=False)
    img_plt_figure = create_top_activating_image_grid(
        activations=None,
        sae_id=sae_dimension,
        dataset=dataset,
        num_rows=2,
        num_cols=5,
        mode="return",
        filter_threshold=None,
        top_images=(top_activating_images, top_activating_values)
    )
    # turn into PIL Image
    import io
    buf = io.BytesIO()
    img_plt_figure.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # close the figure to free memory
    plt.close()
    return img



def get_interpretability_metrics(sae_dimension):
    """Get interpretability metrics for a given SAE dimension."""
    df = load_interpretability_data()
    
    if df.empty:
        return "Interpretability data not available.", "No data available.", "No data available."
    
    # Find the row for this dimension using 'sae_id' column
    dimension_data = df[df['sae_id'] == sae_dimension]
    
    if dimension_data.empty:
        return f"No interpretability data found for SAE dimension {sae_dimension}.", "No data available.", "No data available."
    
    row = dimension_data.iloc[0]
    
    # Global metrics
    global_metrics = f"## Feature Metrics\n\n"
    val = row.get('words_dec_col_score')
    global_metrics += f"- **Words Decoder Similarity**: {f'{val:.4f}' if pd.notna(val) else 'N/A'}\n"
    val = row.get('words_avg_emb_score')
    global_metrics += f"- **Words Avg-Emb Similarity**: {f'{val:.4f}' if pd.notna(val) else 'N/A'}\n"
    val = row.get('ms_score')
    global_metrics += f"- **MS Score**: {f'{val:.4f}' if pd.notna(val) else 'N/A'}\n"
    val = row.get('max_activation')
    global_metrics += f"- **Max Activation**: {f'{val:.4f}' if pd.notna(val) else 'N/A'}\n"
    val = row.get('clarity_score')
    global_metrics += f"- **Clarity Score**: {f'{val:.4f}' if pd.notna(val) else 'N/A'}\n"
    
    # Average Embedding Label Table
    avg_emb_text = "## Average Embedding Suggestions\n\n"
    avg_name = row.get('words_avg_emb_label')
    avg_score = row.get('words_avg_emb_score')
    if pd.notna(avg_name):
        avg_score_str = f"{avg_score:.4f}" if pd.notna(avg_score) else "N/A"
        avg_emb_text += "| Label | Score | Strategy |\n"
        avg_emb_text += "|-------|-------|----------|\n"
        avg_emb_text += f"| {avg_name} | {avg_score_str} | words_avg_emb |\n"
    else:
        avg_emb_text += "No average embedding suggestions available.\n"
    
    # Decoder Label Table
    dec_col_text = "## Decoder Column Suggestions\n\n"
    dec_name = row.get('words_dec_col_label')
    dec_score = row.get('words_dec_col_score')
    if pd.notna(dec_name):
        dec_score_str = f"{dec_score:.4f}" if pd.notna(dec_score) else "N/A"
        dec_col_text += "| Label | Score | Strategy |\n"
        dec_col_text += "|-------|-------|----------|\n"
        dec_col_text += f"| {dec_name} | {dec_score_str} | words_dec_col |\n"
    else:
        dec_col_text += "No decoder column suggestions available.\n"
    
    return global_metrics, avg_emb_text, dec_col_text

def update_dimension_display(sae_dimension):
    """Update the display for a selected SAE dimension."""
    # image = get_top_activating_image_grid(sae_dimension)
    # image = get_top_activating_image(sae_dimension)
    image = get_top_activating_image_grid_on_the_fly(sae_dimension)
    global_metrics, avg_emb_metrics, dec_col_metrics = get_interpretability_metrics(sae_dimension)
    
    return image, global_metrics, avg_emb_metrics, dec_col_metrics

# Create Gradio interface
with gr.Blocks(title="SAE Dimension Interpretability Explorer") as demo:
    gr.Markdown("# SAE Dimension Interpretability Explorer")
    gr.Markdown("Select an SAE dimension to view its top activating images and interpretability metrics.")
    
    with gr.Row():
        dimension_input = gr.Dropdown(
            choices=SAE_DIMENSIONS_LIST,
            value=0,
            label="SAE Dimension",
            info="Select a dimension to explore"
        )
    
    # First row: Image display
    with gr.Row():
        image_display = gr.Image(
            label="Top Activating Images",
            show_label=True,
            height=400
        )
    
    # Second row: Three columns of metrics
    with gr.Row():
        with gr.Column():
            global_metrics_display = gr.Markdown(
                value="Select a dimension to view global metrics.",
                label="Global Metrics"
            )
        
        with gr.Column():
            avg_emb_metrics_display = gr.Markdown(
                value="Select a dimension to view average embedding suggestions.",
                label="Average Embedding Suggestions"
            )
        
        with gr.Column():
            dec_col_metrics_display = gr.Markdown(
                value="Select a dimension to view decoder column suggestions.",
                label="Decoder Column Suggestions"
            )
    
    # Update display when dimension is changed
    dimension_input.change(
        fn=update_dimension_display,
        inputs=[dimension_input],
        outputs=[image_display, global_metrics_display, avg_emb_metrics_display, dec_col_metrics_display]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=6006)
