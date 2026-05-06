import torch
from tqdm import tqdm
import clip
import os
import numpy as np
from path_hub import PathBuilder

def parse_args():
    """
    Parse command line arguments for the script.
    """
    import argparse
    default_output_base_path = PathBuilder().get_shared_vocab_path()
    parser = argparse.ArgumentParser(description="Precompute CLIP embeddings for a vocabulary of words.")
    parser.add_argument('--output_base_path', type=str, default=default_output_base_path,
                        help='Path to save the precomputed CLIP embeddings.')
    parser.add_argument('--clip_model_name', type=str, default='ViT-L/14',
                        help='Name of the CLIP model to use (e.g., ViT-B/16, RN50).')
    parser.add_argument('--vocab_folder', type=str, default=default_output_base_path,
                        help='Path to the folder containing vocabulary files.')
    parser.add_argument('--vocab_name', type=str, required=True,
                        help='Name of the vocabulary file containing words.')
    return parser.parse_args()

def main():
    """
    Main function to precompute CLIP embeddings for a vocabulary of words.
    Saves the embeddings to a specified output path.
    """
    args = parse_args()
    output_base_path = args.output_base_path
    clip_model_name = args.clip_model_name
    vocab_folder = args.vocab_folder
    vocab_name = args.vocab_name

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_base_path), exist_ok=True)
    vocab_path = os.path.join(vocab_folder, vocab_name + ".txt")
    batch_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP model
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    # Tokenize all words
    all_embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(words), batch_size), desc="Encoding text"):
            batch_words = words[i:i+batch_size]
            tokens = clip.tokenize(batch_words).to(device)
            features = model.encode_text(tokens)
            all_embeddings.append(features.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)  # (num_words, embedding_dim)
    
    # output_path
    filename = f"embeddings_clip_{clip_model_name.replace('/', '')}_{vocab_name}_{all_embeddings.shape[0]}_{all_embeddings.shape[1]}.npy"
    output_path= os.path.join(output_base_path, filename)

    # save as memmap
    all_embeddings = all_embeddings.numpy()  # Convert to numpy array
    all_embeddings_memmap = np.memmap(output_path, dtype='float32', mode='w+', shape=all_embeddings.shape)
    all_embeddings_memmap[:] = all_embeddings  # Write data to memmap
    all_embeddings_memmap.flush()  # Flush changes to disk



    print(f"Saved CLIP embeddings to {output_path}")
    print(f"Vocabulary size: {len(words)}, Embedding shape: {all_embeddings.shape}")

if __name__ == "__main__":
    main()