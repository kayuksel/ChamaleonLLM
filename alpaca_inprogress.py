#!/usr/bin/env python
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

import evaluate as hf_evaluate

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
    parser = argparse.ArgumentParser(
        description="Fine-tune a causal LM with LoRA variants for instruction tuning on the Alpaca dataset."
    )
    parser.add_argument("--lm_model_name", type=str, default="gpt2",
                        help="Name of the causal LM from HF (default: 'gpt2')")
    parser.add_argument("--dataset_name", type=str, default="yahma/alpaca-cleaned",
                        help="Dataset name on Hugging Face (default: 'yahma/alpaca-cleaned')")
    parser.add_argument("--dataset_config", type=str, default=None,
                        help="Dataset configuration if any (default: None)")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field name in the dataset containing text (default: 'text')")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization (default: 512)")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--rank", type=int, default=4,
                        help="LoRA rank (default: 4)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Scaling factor for LoRA (default: 1.0)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA (default: 2e-4)")
    # Single experiments argument: comma separated list.
    parser.add_argument("--experiments", type=str, default="unadapted,default,hyper_task,hyper_taskinput",
                        help="Comma-separated list of experiments to run. Options: unadapted, default, hyper_task, hyper_taskinput")
    args = parser.parse_args()
    return args

# ======================================================
# 2. Data Loading and Tokenization for Alpaca
# ======================================================
def format_alpaca_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output_text = example["output"]
    if input_text.strip():
        prompt = (
            f"### Instruction:\n{instruction}\n"
            f"### Input:\n{input_text}\n"
            f"### Response:\n{output_text}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n"
            f"### Response:\n{output_text}"
        )
    return prompt

def tokenize_and_compute_embeddings(example, tokenizer, lm_model, max_length, device):
    # Build the full prompt.
    full_prompt = format_alpaca_prompt(example)
    encoding = tokenizer(full_prompt, truncation=True, max_length=max_length, padding="max_length")
    
    # Identify response marker ("### Response:") and set labels.
    response_marker = "### Response:"
    marker_ids = tokenizer(response_marker, add_special_tokens=False).input_ids
    input_ids = encoding["input_ids"]
    response_start = None
    for i in range(len(input_ids) - len(marker_ids) + 1):
        if input_ids[i:i+len(marker_ids)] == marker_ids:
            response_start = i + len(marker_ids)
            break
    if response_start is not None:
        labels = [-100] * response_start + input_ids[response_start:]
    else:
        labels = input_ids.copy()
    encoding["labels"] = labels

    # Compute the full prompt embedding.
    input_ids_tensor = torch.tensor(encoding["input_ids"]).unsqueeze(0).to(device)
    lm_model.to(device)
    lm_model.eval()
    with torch.no_grad():
        wte = lm_model.transformer.wte.weight
        wpe = lm_model.transformer.wpe.weight
        seq_len = input_ids_tensor.shape[1]
        token_emb = wte[input_ids_tensor]
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = wpe[position_ids]
        full_emb = token_emb + pos_emb
        lm_input_embedding = full_emb.mean(dim=1).squeeze(0)
        encoding["lm_input_embedding"] = lm_input_embedding.cpu().numpy().tolist()
    
    # For clustering, always use instruction+input (if available).
    instruction = example["instruction"]
    input_text = example.get("input", "").strip()
    clustering_text = instruction + "\n" + input_text if input_text else instruction
    clustering_encoding = tokenizer(clustering_text, truncation=True, max_length=max_length, padding="max_length")
    clustering_ids = clustering_encoding["input_ids"]
    clustering_ids_tensor = torch.tensor(clustering_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        clustering_token_emb = lm_model.transformer.wte(clustering_ids_tensor)
        clustering_emb = clustering_token_emb.mean(dim=1).squeeze(0)
        norm = clustering_emb.norm() + 1e-8
        clustering_embedding = (clustering_emb / norm)
        encoding["clustering_embedding"] = clustering_embedding.cpu().numpy().tolist()
    
    # Hyper-LoRA conditioning embeddings.
    # (a) Task-only: use instruction only.
    hyper_task_only_encoding = tokenizer(instruction, truncation=True, max_length=max_length, padding="max_length")
    hyper_task_only_ids = hyper_task_only_encoding["input_ids"]
    hyper_task_only_ids_tensor = torch.tensor(hyper_task_only_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        hyper_task_only_token_emb = lm_model.transformer.wte(hyper_task_only_ids_tensor)
        hyper_task_only_emb = hyper_task_only_token_emb.mean(dim=1).squeeze(0)
        norm = hyper_task_only_emb.norm() + 1e-8
        encoding["hyper_task_token_embedding_task"] = (hyper_task_only_emb / norm).cpu().numpy().tolist()
    
    # (b) Task+input: use clustering_text.
    hyper_task_input_encoding = tokenizer(clustering_text, truncation=True, max_length=max_length, padding="max_length")
    hyper_task_input_ids = hyper_task_input_encoding["input_ids"]
    hyper_task_input_ids_tensor = torch.tensor(hyper_task_input_ids).unsqueeze(0).to(device)
    with torch.no_grad():
        hyper_task_input_token_emb = lm_model.transformer.wte(hyper_task_input_ids_tensor)
        hyper_task_input_emb = hyper_task_input_token_emb.mean(dim=1).squeeze(0)
        norm = hyper_task_input_emb.norm() + 1e-8
        encoding["hyper_task_token_embedding_taskinput"] = (hyper_task_input_emb / norm).cpu().numpy().tolist()
    
    return encoding

def load_and_tokenize_alpaca_dataset(tokenizer, lm_model, max_length, split="train", dataset_name="yahma/alpaca-cleaned", device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    if split == "validation":
        ds_full = load_dataset(dataset_name, split="train")
        ds = ds_full.train_test_split(test_size=0.1, seed=42)["test"]
    else:
        ds = load_dataset(dataset_name, split=split)
    ds = ds.map(lambda ex: tokenize_and_compute_embeddings(ex, tokenizer, lm_model, max_length, device), batched=False)
    return ds

# For non-Alpaca datasets (if needed)
def load_and_tokenize_dataset(tokenizer, max_length=128, split="train", text_field="text", dataset_name="wikitext", dataset_config="wikitext-2-raw-v1"):
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

def precompute_lm_input_embeddings(dataset, lm_model, tokenizer, device):
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
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
    lm_model.to(device)
    lm_model.eval()
    new_embs = []
    with torch.no_grad():
        wte = lm_model.transformer.wte.weight
        for example in tqdm(dataset, desc="Precomputing task token embeddings for clustering/hypernetwork"):
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            token_emb = wte(input_ids)
            mean_emb = token_emb.mean(dim=1).squeeze(0)
            norm = mean_emb.norm() + 1e-8
            normalized_emb = (mean_emb / norm).cpu().numpy().tolist()
            new_embs.append(normalized_emb)
    dataset = dataset.add_column("token_embedding", new_embs)
    return dataset

# ======================================================
# 3. Clustering and DataLoader Functions
# ======================================================
def kmeans_clustering(embeddings, k, num_iters=10):
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

def create_clustered_dataloader(dataset, batch_size, clustering_field):
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
    loader = DataLoader(wrapped_ds, batch_sampler=all_batches, collate_fn=collate_fn)
    print(f"Created DataLoader with {len(all_batches)} batches (clustering on '{clustering_field}').")
    return loader

def save_or_load_dataloader(dataset, batch_size, clustering_field, split_name, dataset_name_arg):
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
# 4. Collate Function
# ======================================================
def collate_fn(batch):
    input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex["labels"] for ex in batch], dtype=torch.long)
    attention_mask = (input_ids != 0).long()
    lm_input_embedding = torch.tensor([ex["lm_input_embedding"] for ex in batch], dtype=torch.float)
    clustering_embedding = torch.tensor([ex["clustering_embedding"] for ex in batch], dtype=torch.float)
    hyper_token_embedding_task = torch.tensor([ex["hyper_task_token_embedding_task"] for ex in batch], dtype=torch.float)
    hyper_token_embedding_taskinput = torch.tensor([ex["hyper_task_token_embedding_taskinput"] for ex in batch], dtype=torch.float)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "lm_input_embedding": lm_input_embedding,
        "clustering_embedding": clustering_embedding,
        "hyper_task_token_embedding_task": hyper_token_embedding_task,
        "hyper_task_token_embedding_taskinput": hyper_token_embedding_taskinput
    }

# ======================================================
# 5. LoRA Modules
# ======================================================
class LoRALinearDefault(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None
        self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        self.B = nn.Parameter(torch.randn(self.out_features, r) * 0.01)
    def forward(self, x):
        original = F.linear(x, self.weight, self.bias)
        lora_update = F.linear(x, self.A)
        lora_update = F.linear(lora_update, self.B)
        return original + self.alpha * lora_update

class LoRALinearHyper(nn.Module):
    def __init__(self, original_linear: nn.Linear, encoder_emb_dim: int, r=4, alpha=1.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.register_buffer("weight", original_linear.weight.detach())
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.detach())
        else:
            self.bias = None
        self.num_params = r * self.in_features + self.out_features * r
        self.hyper_net = nn.Sequential(
            nn.Linear(encoder_emb_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, self.num_params)
        )
    def forward(self, x, token_embeddings=None):
        original_shape = x.shape
        if len(original_shape) == 3:
            B, T, H = x.shape
            x_flat = x.view(B * T, H)
            flat_input = True
        else:
            x_flat = x
            flat_input = False
        if token_embeddings is None:
            token_embeddings = x.mean(dim=1) if len(original_shape) == 3 else x
        mean_encoder = token_embeddings.mean(dim=0, keepdim=True)
        pred = self.hyper_net(mean_encoder).squeeze(0)
        A_flat = pred[: self.r * self.in_features]
        B_flat = pred[self.r * self.in_features:]
        A = A_flat.view(self.r, self.in_features)
        B = B_flat.view(self.out_features, self.r)
        original = F.linear(x_flat, self.weight, self.bias)
        lora_update = F.linear(x_flat, A)
        lora_update = F.linear(lora_update, B)
        out_flat = original + self.alpha * lora_update
        if flat_input:
            out = out_flat.view(original_shape[0], original_shape[1], self.out_features)
        else:
            out = out_flat
        return out

def apply_lora_to_module(module, r=4, alpha=1.0, exclude_types=(nn.Embedding,)):
    for name, child in list(module.named_children()):
        if isinstance(child, exclude_types):
            continue
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinearDefault(child, r=r, alpha=alpha))
        else:
            apply_lora_to_module(child, r=r, alpha=alpha, exclude_types=exclude_types)

def wrap_lm_with_lora(model, method="default", encoder_emb_dim=None, r=4, alpha=1.0):
    # Freeze all original parameters.
    for param in model.parameters():
        param.requires_grad = False
    apply_lora_to_module(model.transformer, r=r, alpha=alpha, exclude_types=(nn.Embedding,))
    if method == "default":
        model.lm_head = LoRALinearDefault(model.lm_head, r=r, alpha=alpha)
        for param in model.lm_head.parameters():
            param.requires_grad = True
    elif method == "hyper":
        if encoder_emb_dim is None:
            raise ValueError("encoder_emb_dim must be provided for hyper method")
        model.lm_head = LoRALinearHyper(model.lm_head, encoder_emb_dim=encoder_emb_dim, r=r, alpha=alpha)
        for param in model.lm_head.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Unknown method")
    return model

# ======================================================
# 6. Training and Evaluation Functions
# ======================================================
def evaluate(model, dataloader, method="default", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), hyper_embedding_field=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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
                token_emb = batch[hyper_embedding_field].to(device) if hyper_embedding_field else batch["hyper_task_token_embedding_taskinput"].to(device)
                transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = transformer_out.last_hidden_state
                logits = model.lm_head(hidden_states, token_embeddings=token_emb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                raise ValueError("Unknown method")
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    model.train()
    return avg_loss

def train_lm(model, train_loader, val_loader, method="default", num_epochs=10, learning_rate=2e-4, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), csv_filename="learning_curve.csv", hyper_embedding_field=None):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
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
            optimizer.zero_grad()
            if method == "default":
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            elif method == "hyper":
                token_emb = batch[hyper_embedding_field].to(device) if hyper_embedding_field else batch["hyper_task_token_embedding_taskinput"].to(device)
                transformer_out = model.transformer(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = transformer_out.last_hidden_state
                logits = model.lm_head(hidden_states, token_embeddings=token_emb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                raise ValueError("Unknown method")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix(loss=total_loss/num_batches)
        avg_train_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        val_loss = evaluate(model, val_loader, method=method, device=device, hyper_embedding_field=hyper_embedding_field)
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
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["epoch", "train_loss", "val_loss", "val_perplexity"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            writer.writerow(record)
    print(f"Learning curves saved to {csv_filename}")
    return best_val_loss, history

# ======================================================
# 7. Utility to Count Trainable Parameters
# ======================================================
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ======================================================
# 8. Generation and Metric Evaluation Functions
# ======================================================
def format_alpaca_prompt_for_generation(example):
    instruction = example["instruction"]
    input_text = example.get("input", "").strip()
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n### Input:\n{input_text}\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n### Response:\n"
    return prompt

def compute_generation_metrics(model, dataset, tokenizer, device, max_new_tokens=100):
    model.eval()
    predictions = []
    references = []
    for example in tqdm(dataset, desc="Generating predictions"):
        prompt = format_alpaca_prompt_for_generation(example)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        input_length = inputs["input_ids"].shape[1]
        generated_ids = generated_ids[0][input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predictions.append(generated_text)
        references.append(example["output"].strip())
    bleu_metric = hf_evaluate.load("bleu")
    meteor_metric = hf_evaluate.load("meteor")
    rouge_metric = hf_evaluate.load("rouge")
    bleu_result = bleu_metric.compute(predictions=predictions, references=[[ref] for ref in references])
    meteor_result = meteor_metric.compute(predictions=predictions, references=references)
    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    return bleu_result, meteor_result, rouge_result

# ======================================================
# 9. Main: Putting Everything Together
# ======================================================
def main():
    args = parse_args()
    # Split the single experiments argument into a list.
    experiments = [e.strip().lower() for e in args.experiments.split(",")]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer and base model.
    tokenizer = AutoTokenizer.from_pretrained(args.lm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading pre-trained LM...")
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
    lm_model.config.pad_token_id = tokenizer.pad_token_id
    total_params = sum(p.numel() for p in lm_model.parameters())
    print(f"Total parameters in unadapted LM: {total_params}")

    # Load dataset.
    if "alpaca" in args.dataset_name.lower():
        print("Loading and tokenizing Alpaca dataset...")
        train_ds = load_and_tokenize_alpaca_dataset(tokenizer, lm_model, max_length=args.max_length, split="train", dataset_name=args.dataset_name, device=DEVICE)
        val_ds = load_and_tokenize_alpaca_dataset(tokenizer, lm_model, max_length=args.max_length, split="validation", dataset_name=args.dataset_name, device=DEVICE)
        # For clustering, always use instruction+input.
        clustering_field = "clustering_embedding"
        encoder_emb_dim = len(train_ds[0]["hyper_task_token_embedding_taskinput"])
    else:
        print("Loading and tokenizing dataset...")
        train_ds = load_and_tokenize_dataset(tokenizer, max_length=args.max_length, split="train",
                                             text_field=args.text_field, dataset_name=args.dataset_name, dataset_config=args.dataset_config)
        val_ds = load_and_tokenize_dataset(tokenizer, max_length=args.max_length, split="validation",
                                           text_field=args.text_field, dataset_name=args.dataset_name, dataset_config=args.dataset_config)
        print("Precomputing LM input embeddings for training set...")
        train_ds = precompute_lm_input_embeddings(train_ds, lm_model, tokenizer, DEVICE)
        print("Precomputing LM input embeddings for validation set...")
        val_ds = precompute_lm_input_embeddings(val_ds, lm_model, tokenizer, DEVICE)
        print("Precomputing token embeddings for clustering/hypernetwork (training set)...")
        train_ds = precompute_token_embeddings(train_ds, lm_model, tokenizer, DEVICE)
        print("Precomputing token embeddings for clustering/hypernetwork (validation set)...")
        val_ds = precompute_token_embeddings(val_ds, lm_model, tokenizer, DEVICE)
        clustering_field = "token_embedding"
        encoder_emb_dim = len(train_ds[0]["token_embedding"])

    ds_name_for_pickle = args.dataset_name.replace("/", "_")
    train_loader = save_or_load_dataloader(train_ds, batch_size=args.batch_size, clustering_field=clustering_field, split_name="train", dataset_name_arg=ds_name_for_pickle)
    val_loader = save_or_load_dataloader(val_ds, batch_size=args.batch_size, clustering_field=clustering_field, split_name="val", dataset_name_arg=ds_name_for_pickle)

    # --------------------------
    # Run experiments as specified.
    # --------------------------
    # 1. Unadapted GPT-2.
    if "unadapted" in experiments:
        print("\n=== Running Unadapted GPT-2 Experiment ===")
        unadapted_model = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
        unadapted_model.to(DEVICE)
        val_loss = evaluate(unadapted_model, val_loader, method="default", device=DEVICE)
        val_perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")
        print(f"Unadapted GPT-2: Val Loss = {val_loss:.4f} | Val Perplexity = {val_perplexity:.4f}")
        bleu, meteor, rouge = compute_generation_metrics(unadapted_model, val_ds, tokenizer, DEVICE, max_new_tokens=100)
        print("Unadapted GPT-2 Generation Metrics:")
        print("BLEU:", bleu)
        print("METEOR:", meteor)
        print("ROUGE:", rouge)

    # 2. Default LoRA.
    if "default" in experiments:
        print("\n=== Running Default LoRA Experiment ===")
        model_default = wrap_lm_with_lora(lm_model, method="default", r=args.rank, alpha=args.alpha)
        print(f"Default LoRA Trainable Parameters: {count_trainable_params(model_default)}")
        best_val_loss_default, _ = train_lm(
            model_default, train_loader, val_loader, method="default",
            num_epochs=args.num_epochs, learning_rate=args.lr, device=DEVICE,
            csv_filename="learning_curve_default.csv"
        )
        print(f"Default LoRA: Best Validation Loss = {best_val_loss_default:.4f}")
        bleu, meteor, rouge = compute_generation_metrics(model_default, val_ds, tokenizer, DEVICE, max_new_tokens=100)
        print("Default LoRA Generation Metrics:")
        print("BLEU:", bleu)
        print("METEOR:", meteor)
        print("ROUGE:", rouge)

    # 3. Hyper LoRA experiments.
    hyper_experiments = []
    if "hyper_task" in experiments:
        hyper_experiments.append("task")
    if "hyper_taskinput" in experiments:
        hyper_experiments.append("taskinput")
    hyper_models = {}
    for mode in hyper_experiments:
        print(f"\n--- Running Hyper LoRA Experiment using {mode} hyper embeddings ---")
        hyper_model_instance = AutoModelForCausalLM.from_pretrained(args.lm_model_name)
        if hyper_model_instance.config.pad_token_id is None:
            hyper_model_instance.config.pad_token_id = tokenizer.pad_token_id
        model_hyper = wrap_lm_with_lora(
            hyper_model_instance,
            method="hyper",
            encoder_emb_dim=encoder_emb_dim,
            r=args.rank,
            alpha=args.alpha
        )
        print(f"Hyper LoRA ({mode}) Trainable Parameters: {count_trainable_params(model_hyper)}")
        best_val_loss, _ = train_lm(
            model_hyper, train_loader, val_loader, method="hyper",
            num_epochs=args.num_epochs, learning_rate=args.lr, device=DEVICE,
            csv_filename=f"learning_curve_hyper_{mode}.csv",
            hyper_embedding_field=f"hyper_task_token_embedding_{mode}"
        )
        print(f"Hyper LoRA ({mode}): Best Validation Loss = {best_val_loss:.4f}")
        hyper_models[mode] = model_hyper

    for mode, model in hyper_models.items():
        print(f"\n=== Generating Metrics for Hyper LoRA ({mode}) Model ===")
        bleu, meteor, rouge = compute_generation_metrics(model, val_ds, tokenizer, DEVICE, max_new_tokens=100)
        print(f"Hyper LoRA ({mode}) Generation Metrics:")
        print("BLEU:", bleu)
        print("METEOR:", meteor)
        print("ROUGE:", rouge)

if __name__ == "__main__":
    main()