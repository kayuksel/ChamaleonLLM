import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle as cPickle
import csv
import math

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# ======================================================
# Global Dataset Wrapper for Pickling
# ======================================================

class HFWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        return self.dataset[idx]

# ======================================================
# 1. Configuration via Arguments
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with hyper‑network LoRA.")
    parser.add_argument("--lm_model_name", type=str, default="gpt2",
                        help="Name of the causal LM from HF (default: 'gpt2')")
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Dataset name (default: 'wikitext')")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="Dataset configuration (default: 'wikitext-2-raw-v1')")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field name in the dataset containing the text (default: 'text')")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum sequence length for tokenization (default: 256)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank (default: 4)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling factor for LoRA (default: 1.0)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Learning rate for LoRA (default: 5e-4)")
    args = parser.parse_args()
    return args

# ======================================================
# 2. Load and Tokenize Dataset
# ======================================================

def load_and_tokenize_dataset(tokenizer, max_length=128, split="train", text_field="text", 
                               dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
    """
    Loads and tokenizes the dataset.
    Returns a HF Dataset with fields:
      - input_ids: tokenized ids.
      - labels: a copy of input_ids.
      - raw_text: the original text.
    """
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.filter(lambda ex: ex[text_field] is not None and len(ex[text_field]) > 0)
    def tokenize_fn(example):
        enc = tokenizer(
            example[text_field],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
        enc["labels"] = enc["input_ids"].copy()
        enc["raw_text"] = example[text_field]
        return enc
    ds = ds.map(tokenize_fn, batched=False)
    return ds

# ======================================================
# 3. Precompute Embeddings
# ======================================================

def precompute_lm_input_embeddings(dataset, lm_model, tokenizer, device):
    """
    Compute the LM input embeddings (token + positional embeddings averaged over the sequence)
    for each example. Stored as "lm_input_embedding".
    """
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
        # For GPT-2: use transformer.wte (token embeddings) and transformer.wpe (positional embeddings)
        wte = lm_model.transformer.wte.weight
        wpe = lm_model.transformer.wpe.weight
        for example in tqdm(dataset, desc="Precomputing LM input embeddings"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            seq_len = input_ids.shape[1]
            token_emb = wte[input_ids]
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = wpe[position_ids]
            input_emb = token_emb + pos_emb
            mean_emb = input_emb.mean(dim=1).squeeze(0)
            new_embs.append(mean_emb.cpu().numpy().tolist())
    dataset = dataset.add_column("lm_input_embedding", new_embs)
    return dataset

def precompute_token_embeddings(dataset, lm_model, tokenizer, device):
    """
    Compute the average token embedding (ignoring positional embeddings) for each example.
    L2-normalized and stored as "token_embedding".
    """
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
        wte = lm_model.transformer.wte.weight
        for example in tqdm(dataset, desc="Precomputing token embeddings for clustering/hypernetwork"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            token_emb = wte[input_ids]
            mean_emb = token_emb.mean(dim=1).squeeze(0)
            norm = mean_emb.norm() + 1e-8
            normalized_emb = mean_emb / norm
            new_embs.append(normalized_emb.cpu().numpy().tolist())
    dataset = dataset.add_column("token_embedding", new_embs)
    return dataset

# ======================================================
# 4. Clustering and DataLoader
# ======================================================

def kmeans_plus_plus_init(embeddings, k, similarity_metric="cosine"):
    """
    K-Means++ Initialization for better centroid selection.
    
    Args:
        embeddings (torch.Tensor): (N, D) token embeddings.
        k (int): Number of clusters.
        similarity_metric (str): "cosine" or "euclidean" for distance calculation.
    
    Returns:
        torch.Tensor: (k, D) initialized cluster centroids.
    """
    N, D = embeddings.shape
    centroids = torch.empty((k, D), dtype=torch.float, device=embeddings.device)
    first_idx = torch.randint(0, N, (1,))
    centroids[0] = embeddings[first_idx]

    for j in range(1, k):
        if similarity_metric == "cosine":
            dists = 1 - torch.mm(embeddings, centroids[:j].T)  # Cosine distance: 1 - cosine_similarity
        else:  # Euclidean
            dists = torch.cdist(embeddings, centroids[:j], p=2).min(dim=1)[0] ** 2  # Squared L2 distance

        prob = dists / dists.sum()
        new_idx = torch.multinomial(prob, 1)
        centroids[j] = embeddings[new_idx]

    return centroids

def kmeans_clustering(embeddings, k, num_iters=20, tol=1e-4, similarity_metric="cosine"):
    """
    K-Means Clustering with configurable similarity metric, K-Means++ initialization, and Early Stopping.

    Args:
        embeddings (torch.Tensor): (N, D) token embeddings.
        k (int): Number of clusters.
        num_iters (int): Maximum iterations.
        tol (float): Threshold for early stopping.
        similarity_metric (str): "cosine" or "euclidean" for distance calculation.

    Returns:
        torch.Tensor: Cluster assignments (N,)
    """
    if similarity_metric == "cosine":
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    centroids = kmeans_plus_plus_init(embeddings, k, similarity_metric)
    prev_centroids = torch.zeros_like(centroids)

    for i in range(num_iters):
        if similarity_metric == "cosine":
            dists = 1 - torch.mm(embeddings, centroids.T)  # Cosine distance: 1 - cos_sim
        else:  # Euclidean
            dists = torch.cdist(embeddings, centroids, p=2)  # L2 (Euclidean) distance

        assignments = torch.argmin(dists, dim=1)

        new_centroids = torch.stack([
            embeddings[assignments == j].mean(dim=0) if (assignments == j).sum() > 0 else centroids[j]
            for j in range(k)
        ])

        # Early stopping
        centroid_shift = torch.norm(new_centroids - prev_centroids, p=2)
        if centroid_shift < tol:
            print(f"Early stopping at iteration {i+1} (centroid shift = {centroid_shift:.6f})")
            break

        prev_centroids = new_centroids
        centroids = new_centroids

    return assignments


def create_clustered_dataloader(dataset, batch_size, clustering_field="token_embedding"):
    """
    Cluster examples based on the specified field and create a DataLoader.
    """
    N = len(dataset)
    num_clusters = max(1, N // batch_size)
    all_emb = torch.tensor([ex[clustering_field] for ex in dataset], dtype=torch.float)
    assignments = kmeans_clustering(all_emb, num_clusters)
    cluster_to_indices = {}
    for idx, cluster in enumerate(assignments.tolist()):
        cluster_to_indices.setdefault(cluster, []).append(idx)
    all_batches = []
    for indices in cluster_to_indices.values():
        np.random.shuffle(indices)
        for i in range(0, len(indices), batch_size):
            all_batches.append(indices[i:i+batch_size])
    wrapped_ds = HFWrapper(dataset)
    # Use the list of batches as the batch_sampler.
    loader = DataLoader(wrapped_ds, batch_sampler=all_batches, collate_fn=collate_fn)
    print(f"Created DataLoader with {len(all_batches)} batches (clustering on '{clustering_field}').")
    return loader

def save_or_load_dataloader(dataset, batch_size, clustering_field, split_name, dataset_name_arg):
    """
    Checks if a pickle file for the DataLoader exists.
    If so, loads and returns it. Otherwise, creates the DataLoader,
    saves it to a pickle file, and returns it.
    """
    pickle_filename = f"{dataset_name_arg}_{split_name}_loader.pkl"
    if os.path.exists(pickle_filename):
        print(f"Loading {split_name} DataLoader from {pickle_filename} ...")
        with open(pickle_filename, "rb") as f:
            loader = cPickle.load(f)
    else:
        print(f"Creating {split_name} DataLoader ...")
        loader = create_clustered_dataloader(dataset, batch_size, clustering_field)
        print(f"Saving {split_name} DataLoader to {pickle_filename} ...")
        with open(pickle_filename, "wb") as f:
            cPickle.dump(loader, f, protocol=cPickle.HIGHEST_PROTOCOL)
    return loader

# ======================================================
# 5. Collate Function
# ======================================================

def collate_fn(batch):
    """
    Collate function that returns:
      - input_ids, labels, attention_mask, lm_input_embedding, and token_embedding.
    """
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    attention_mask = (input_ids != 0).long()
    lm_input_embedding = torch.tensor([ex["lm_input_embedding"] for ex in batch], dtype=torch.float)
    token_embedding = torch.tensor([ex["token_embedding"] for ex in batch], dtype=torch.float)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "lm_input_embedding": lm_input_embedding,
        "token_embedding": token_embedding
    }

# ======================================================
# 6. LoRA Modules
# ======================================================

class LoRALinearDefault(nn.Module):
    """
    A generic LoRA wrapper for any nn.Linear module.
    Freezes the original weight (and bias, if present) and adds a trainable low‑rank update.
    Used for adapting transformer layers.
    """
    def __init__(self, original_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        # Freeze original weights.
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None
        # Initialize LoRA parameters.
        self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)

    def forward(self, x):
        original = F.linear(x, self.weight, self.bias)
        lora_update = F.linear(x, self.A)
        lora_update = F.linear(lora_update, self.B)
        return original + self.alpha * lora_update

class LoRALinearHyper(nn.Module):
    """
    A LoRA wrapper for an nn.Linear layer that uses a hyper‑network to predict
    the low‑rank update parameters from an external embedding.
    Used for adapting the LM head.
    """
    def __init__(self, original_linear: nn.Linear, encoder_emb_dim: int, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Freeze original weights.
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None

        # The total number of parameters to predict for the LoRA update.
        self.num_params = r * self.in_features + self.out_features * r
        
        # Define the hyper‑network.
        self.hyper_net = nn.Sequential(
            nn.Linear(encoder_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_params)
        )
        
        # Initialize earlier layers (all except the final one) using standard Xavier initialization.
        for layer in list(self.hyper_net.children())[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        
        # Initialize the final layer with small non‑zero values so that the hyper‑network
        # outputs are near zero but still allow for gradient flow.
        final_layer = self.hyper_net[-1]
        nn.init.normal_(final_layer.weight, mean=0.0, std=1e-3)
        if final_layer.bias is not None:
            nn.init.normal_(final_layer.bias, mean=0.0, std=1e-3)
    
    def forward(self, x, token_embeddings):
        # Compute the mean of the token embeddings to serve as input to the hyper‑network.
        mean_encoder = token_embeddings.mean(dim=0, keepdim=True)  # Shape: [1, encoder_emb_dim]
        pred = self.hyper_net(mean_encoder).squeeze(0)            # Shape: [num_params]
        
        # Split the predicted parameters into matrices A and B.
        A_flat = pred[: self.r * self.in_features]
        B_flat = pred[self.r * self.in_features:]
        A = A_flat.view(self.r, self.in_features)
        B = B_flat.view(self.out_features, self.r)
        
        # Compute the original output from the frozen weight and bias.
        original = F.linear(x, self.weight, self.bias)
        # Compute the LoRA update.
        lora_update = F.linear(x, A)
        lora_update = F.linear(lora_update, B)
        
        # Combine the original output and the scaled LoRA update.
        return original + self.alpha * lora_update


# ======================================================
# 7. Model Wrapping Functions
# ======================================================

def apply_lora_to_module(module, r=4, alpha=1.0, exclude_types=(nn.Embedding,)):
    """
    Recursively replace all nn.Linear modules in 'module' with a default LoRA-wrapped version,
    while skipping modules whose type is in 'exclude_types'.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, exclude_types):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinearDefault(child, r=r, alpha=alpha))
        else:
            apply_lora_to_module(child, r=r, alpha=alpha, exclude_types=exclude_types)

def wrap_lm_with_lora(model, encoder_emb_dim, r=4, alpha=1.0):
    """
    Freeze the LM's original parameters and then apply LoRA fine-tuning:
      - All transformer layers are wrapped with the default LoRA update.
      - The LM head is wrapped with the hyper‑network variant.
    """
    # Freeze all parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to all applicable linear layers in the transformer.
    apply_lora_to_module(model.transformer, r=r, alpha=alpha, exclude_types=(nn.Embedding,))
    
    # Wrap the LM head with the hyper‑network variant.
    model.lm_head = LoRALinearHyper(model.lm_head, encoder_emb_dim=encoder_emb_dim, r=r, alpha=alpha)
    for param in model.lm_head.parameters():
        param.requires_grad = True

    return model

# ======================================================
# 8. Training and Evaluation Loops (Hyper‑network Only)
# ======================================================

def evaluate_hyper(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate the hyper‑network LoRA model on the validation set and return the average loss.
    Uses the hyper‑network logic (passing token_embeddings to the LM head).
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_emb = batch["token_embedding"].to(device)
            # Get transformer output and pass the last hidden state through the hyper LM head.
            transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = transformer_out.last_hidden_state
            last_hidden = hidden_states[:, -1, :]
            logits = model.lm_head(last_hidden, token_embeddings=token_emb)
            target = labels[:, -1]
            loss = loss_fn(logits, target)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    model.train()
    return avg_loss

def train_lm(model, train_loader, val_loader, num_epochs=5, learning_rate=5e-4,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
             csv_filename="learning_curve.csv"):
    """
    Train the LM for next-token prediction using the hyper‑network variant.
    After each epoch, evaluate on the validation set.
    Saves the learning curves (train loss, val loss, and val perplexity) to a CSV file.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float("inf")
    history = []
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_emb = batch["token_embedding"].to(device)
            optimizer.zero_grad()
            transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = transformer_out.last_hidden_state
            last_hidden = hidden_states[:, -1, :]
            logits = model.lm_head(last_hidden, token_embeddings=token_emb)
            target = labels[:, -1]
            loss = loss_fn(logits, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=total_loss / num_batches)
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        val_loss = evaluate_hyper(model, val_loader, device=device)
        val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Perplexity = {val_perplexity:.4f}")
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_perplexity": val_perplexity
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # Save learning curves to CSV.
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_perplexity"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(record)
    print(f"Learning curves saved to {csv_filename}")
    return best_val_loss, history

def evaluate_original(model, dataloader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate an unadapted (original) GPT-2 on the given dataloader using its standard forward pass.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Original Model", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    model.train()
    return avg_loss

# ======================================================
# 9. Utility to Count Trainable Parameters
# ======================================================

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ======================================================
# 10. Main: Putting Everything Together
# ======================================================

def main():
    args = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and LM.
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading pre-trained LM...")
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    
    # Print total number of parameters of the unadapted GPT-2.
    total_params = sum(p.numel() for p in lm_model.parameters())
    print(f"Total parameters in unadapted GPT-2: {total_params}")

    # Load training and validation splits.
    print("Loading and tokenizing training set...")
    train_ds = load_and_tokenize_dataset(tokenizer, max_length=args.max_length, split="train",
                                         text_field=args.text_field, dataset_name=args.dataset_name, dataset_config=args.dataset_config)
    print("Loading and tokenizing validation set...")
    val_ds = load_and_tokenize_dataset(tokenizer, max_length=args.max_length, split="validation",
                                       text_field=args.text_field, dataset_name=args.dataset_name, dataset_config=args.dataset_config)
    
    # Precompute embeddings.
    print("Precomputing LM input embeddings for training set...")
    train_ds = precompute_lm_input_embeddings(train_ds, lm_model, tokenizer, DEVICE)
    print("Precomputing LM input embeddings for validation set...")
    val_ds = precompute_lm_input_embeddings(val_ds, lm_model, tokenizer, DEVICE)
    
    print("Precomputing token embeddings for clustering/hypernetwork (training set)...")
    train_ds = precompute_token_embeddings(train_ds, lm_model, tokenizer, DEVICE)
    print("Precomputing token embeddings for clustering/hypernetwork (validation set)...")
    val_ds = precompute_token_embeddings(val_ds, lm_model, tokenizer, DEVICE)
    
    # Create clustered DataLoaders.
    train_loader = save_or_load_dataloader(train_ds, batch_size=args.batch_size, clustering_field="token_embedding", 
                                           split_name="train", dataset_name_arg=args.dataset_name)
    val_loader = save_or_load_dataloader(val_ds, batch_size=args.batch_size, clustering_field="token_embedding", 
                                         split_name="val", dataset_name_arg=args.dataset_name)

    # Training using Hyper‑network LoRA (Transformer layers use default LoRA).
    encoder_emb_dim = len(train_ds[0]["token_embedding"])
    print("\n=== Training with Hyper LoRA (Transformer + LM Head) ===")
    lm_model_hyper = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    if lm_model_hyper.config.pad_token_id is None:
        lm_model_hyper.config.pad_token_id = tokenizer.pad_token_id
    model_hyper = wrap_lm_with_lora(lm_model_hyper, encoder_emb_dim=encoder_emb_dim,
                                    r=args.rank, alpha=args.alpha)
    print(f"Hyper LoRA Trainable Parameters: {count_trainable_params(model_hyper)}")
    best_val_loss_hyper, history_hyper = train_lm(
        model_hyper, train_loader, val_loader,
        num_epochs=args.num_epochs, learning_rate=args.lr, device=DEVICE,
        csv_filename="learning_curve_hyper.csv"
    )
    print(f"Best Validation Loss (Hyper LoRA): {best_val_loss_hyper:.4f}")

if __name__ == "__main__":
    main()
