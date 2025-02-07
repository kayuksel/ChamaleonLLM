import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle as cPickle  # Using pickle with the highest protocol.
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
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM with LoRA variants.")
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
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
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

def load_and_tokenize_dataset(tokenizer, max_length=128, split="train", text_field="text", dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
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
    for each example. This is stored as "lm_input_embedding" (used by the model/hyper‑network).
    """
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
        # For GPT2: use transformer.wte (token embeddings) and transformer.wpe (positional embeddings)
        wte = lm_model.transformer.wte.weight  # shape: (vocab_size, hidden_size)
        wpe = lm_model.transformer.wpe.weight  # shape: (max_position_embeddings, hidden_size)
        for example in tqdm(dataset, desc="Precomputing LM input embeddings"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)  # shape: (1, seq_len)
            seq_len = input_ids.shape[1]
            token_emb = wte[input_ids]  # (1, seq_len, hidden_size)
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
            pos_emb = wpe[position_ids]  # (1, seq_len, hidden_size)
            input_emb = token_emb + pos_emb  # (1, seq_len, hidden_size)
            mean_emb = input_emb.mean(dim=1).squeeze(0)  # (hidden_size,)
            new_embs.append(mean_emb.cpu().numpy().tolist())
    dataset = dataset.add_column("lm_input_embedding", new_embs)
    return dataset

def precompute_token_embeddings(dataset, lm_model, tokenizer, device):
    """
    Compute the average token embedding (ignoring positional embeddings) for each example.
    This embedding is L2-normalized and stored as "token_embedding".
    It is used both for clustering and (in the hyper‑network version) as input.
    """
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
        wte = lm_model.transformer.wte.weight  # shape: (vocab_size, hidden_size)
        for example in tqdm(dataset, desc="Precomputing token embeddings for clustering/hypernetwork"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)  # (1, seq_len)
            token_emb = wte[input_ids]  # (1, seq_len, hidden_size)
            mean_emb = token_emb.mean(dim=1).squeeze(0)  # (hidden_size,)
            norm = mean_emb.norm() + 1e-8
            normalized_emb = mean_emb / norm
            new_embs.append(normalized_emb.cpu().numpy().tolist())
    dataset = dataset.add_column("token_embedding", new_embs)
    return dataset

# ======================================================
# 4. Clustering and DataLoader
# ======================================================

def kmeans_clustering(embeddings, k, num_iters=10):
    """
    A simple k‑means clustering implementation using Euclidean distance.
    (Embeddings are assumed to be normalized.)
    embeddings: (N, D) torch.Tensor.
    k: number of clusters.
    Returns: assignments (tensor of shape (N,))
    """
    N, D = embeddings.shape
    indices = torch.randperm(N)[:k]
    centroids = embeddings[indices].clone()
    for _ in range(num_iters):
        dists = torch.cdist(embeddings, centroids, p=2) ** 2
        assignments = torch.argmin(dists, dim=1)
        for j in range(k):
            if (assignments == j).sum() > 0:
                centroids[j] = embeddings[assignments == j].mean(dim=0)
    return assignments

def create_clustered_dataloader(dataset, batch_size, clustering_field="token_embedding"):
    """
    Cluster examples based on the specified field (e.g., "token_embedding") and create a DataLoader.
    Both training and validation DataLoaders are built in this way.
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
    The filename uses the provided dataset_name_arg.
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
        
        self.num_params = r * self.in_features + self.out_features * r
        # Define the hyper‑network (you can adjust the architecture as needed).
        self.hyper_net = nn.Sequential(
            nn.Linear(encoder_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_params)
        )
        
    def forward(self, x, token_embeddings):
        # Compute the mean of the token embeddings as input to the hyper‑network.
        mean_encoder = token_embeddings.mean(dim=0, keepdim=True)  # shape: [1, encoder_emb_dim]
        pred = self.hyper_net(mean_encoder).squeeze(0)  # shape: [num_params]
        A_flat = pred[: self.r * self.in_features]
        B_flat = pred[self.r * self.in_features:]
        A = A_flat.view(self.r, self.in_features)
        B = B_flat.view(self.out_features, self.r)
        original = F.linear(x, self.weight, self.bias)
        lora_update = F.linear(x, A)
        lora_update = F.linear(lora_update, B)
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

def wrap_lm_with_lora(model, method="default", encoder_emb_dim=None, r=4, alpha=1.0):
    """
    Freeze the LM's original parameters and then apply LoRA fine-tuning as follows:
      - All transformer layers (attention, feed‑forward, etc.) are wrapped with the default LoRA update.
      - The LM head is wrapped with either the default or hyper‑network variant based on 'method'.
    
    The embeddings (and similar modules) are not wrapped.
    """
    # Freeze all parameters.
    for param in model.parameters():
        param.requires_grad = False

    # Apply default LoRA to all applicable linear layers in the transformer.
    apply_lora_to_module(model.transformer, r=r, alpha=alpha, exclude_types=(nn.Embedding,))
    
    # Wrap the LM head.
    if method == "default":
        model.lm_head = LoRALinearDefault(model.lm_head, r=r, alpha=alpha)
        # Enable training on the LM head's LoRA parameters.
        for param in model.lm_head.parameters():
            param.requires_grad = True
    elif method == "hyper":
        if encoder_emb_dim is None:
            raise ValueError("encoder_emb_dim must be provided for hyper method")
        model.lm_head = LoRALinearHyper(model.lm_head, encoder_emb_dim=encoder_emb_dim, r=r, alpha=alpha)
        # Enable training on the LM head's hyper‑network parameters.
        for param in model.lm_head.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unknown method")
    
    return model

# ======================================================
# 8. Training and Evaluation Loops
# ======================================================

def evaluate(model, dataloader, method="default", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Evaluate the model on the validation set and return the average loss.
    For the hyper method, we assume that the LM head expects an extra token_embedding input.
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
            if method == "default":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif method == "hyper":
                token_emb = batch["token_embedding"].to(device)
                transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = transformer_out.last_hidden_state
                last_hidden = hidden_states[:, -1, :]
                logits = model.lm_head(last_hidden, token_embeddings=token_emb)
                target = labels[:, -1]
                loss = loss_fn(logits, target)
            else:
                raise ValueError("Unknown method")
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    model.train()
    return avg_loss

def train_lm(model, train_loader, val_loader, method="default", num_epochs=10, learning_rate=5e-4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), csv_filename="learning_curve.csv"):
    """
    Train the LM for next-token prediction.
    After each epoch, evaluate on the validation set.
    Saves the learning curves (train loss, val loss, and val perplexity) to a CSV file.
    Returns the best validation loss achieved and the training history.
    """
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float("inf")
    history = []  # To store epoch, train loss, val loss, and perplexity.
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            if method == "default":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif method == "hyper":
                token_emb = batch["token_embedding"].to(device)
                transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = transformer_out.last_hidden_state
                last_hidden = hidden_states[:, -1, :]
                logits = model.lm_head(last_hidden, token_embeddings=token_emb)
                target = labels[:, -1]
                loss = loss_fn(logits, target)
            else:
                raise ValueError("Unknown method")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=total_loss/num_batches)
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        val_loss = evaluate(model, val_loader, method=method, device=device)
        val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")  # avoid overflow
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

    # -----------------------------------------------------
    # If the pickle files for the DataLoaders exist (named using dataset_name),
    # load them directly. Otherwise, download/process the dataset.
    # -----------------------------------------------------
    train_pickle = f"{args.dataset_name}_train_loader.pkl"
    val_pickle = f"{args.dataset_name}_val_loader.pkl"
    if os.path.exists(train_pickle) and os.path.exists(val_pickle):
        print("Pickle files exist. Loading DataLoaders from pickle...")
        with open(train_pickle, "rb") as f:
            train_loader = cPickle.load(f)
        with open(val_pickle, "rb") as f:
            val_loader = cPickle.load(f)
    else:
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
        train_loader = save_or_load_dataloader(train_ds, batch_size=args.batch_size, clustering_field="token_embedding", split_name="train", dataset_name_arg=args.dataset_name)
        val_loader = save_or_load_dataloader(val_ds, batch_size=args.batch_size, clustering_field="token_embedding", split_name="val", dataset_name_arg=args.dataset_name)

    # -----------------------------------------------------
    # Option 1: Use default LoRA for LM head (and default LoRA for all transformer layers).
    print("\n=== Training with Default LoRA (Transformer + LM Head) ===")
    model_default = wrap_lm_with_lora(lm_model, method="default", r=args.rank, alpha=args.alpha)
    print(f"Default LoRA Trainable Parameters: {count_trainable_params(model_default)}")
    best_val_loss_default, history_default = train_lm(
        model_default, train_loader, val_loader, method="default",
        num_epochs=args.num_epochs, learning_rate=args.lr, device=DEVICE,
        csv_filename="learning_curve_default.csv"
    )
    print(f"Best Validation Loss (Default LoRA): {best_val_loss_default:.4f}")
    
    # -----------------------------------------------------
    # Option 2: Use hyper‑network LoRA for LM head.
    encoder_emb_dim = len(train_loader.dataset.dataset[0]["token_embedding"])
    print("\n=== Training with Hyper LoRA (Transformer + LM Head) ===")
    lm_model_hyper = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    if lm_model_hyper.config.pad_token_id is None:
        lm_model_hyper.config.pad_token_id = tokenizer.pad_token_id
    model_hyper = wrap_lm_with_lora(lm_model_hyper, method="hyper", encoder_emb_dim=encoder_emb_dim,
                                    r=args.rank, alpha=args.alpha)
    print(f"Hyper LoRA Trainable Parameters: {count_trainable_params(model_hyper)}")
    best_val_loss_hyper, history_hyper = train_lm(
        model_hyper, train_loader, val_loader, method="hyper",
        num_epochs=args.num_epochs, learning_rate=args.lr, device=DEVICE,
        csv_filename="learning_curve_hyper.csv"
    )
    print(f"Best Validation Loss (Hyper LoRA): {best_val_loss_hyper:.4f}")

    # Compare the two fine-tuning strategies.
    if best_val_loss_default < best_val_loss_hyper:
        print("\nDefault LoRA achieved a better validation loss.")
    elif best_val_loss_default > best_val_loss_hyper:
        print("\nHyper LoRA achieved a better validation loss.")
    else:
        print("\nBoth methods achieved the same best validation loss.")

    # -----------------------------------------------------
    # Evaluate the original (unadapted, frozen) GPT-2 on the validation set.
    print("\n=== Evaluating Unadapted (Frozen) GPT-2 ===")
    orig_model = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    orig_model.to(DEVICE)
    orig_val_loss = evaluate(orig_model, val_loader, method="default", device=DEVICE)
    orig_val_perplexity = math.exp(orig_val_loss) if orig_val_loss < 20 else float("inf")
    print(f"Original GPT-2 Validation Loss: {orig_val_loss:.4f} | Validation Perplexity: {orig_val_perplexity:.4f}")

    orig_val_loss = evaluate(orig_model, train_loader, method="default", device=DEVICE)
    orig_val_perplexity = math.exp(orig_val_loss) if orig_val_loss < 20 else float("inf")
    print(f"Original GPT-2 Train Loss: {orig_val_loss:.4f} | Validation Perplexity: {orig_val_perplexity:.4f}")

if __name__ == "__main__":
    main()
