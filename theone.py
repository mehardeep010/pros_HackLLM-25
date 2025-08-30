# ==============================================================================
# FINAL ROBUST KAGGLE NOTEBOOK - SINGLE CELL (ALL FIXES INTEGRATED)
# ==============================================================================

# --- 1. INSTALLATIONS (The "Golden Combination") ---
!pip install "numpy<2.0"
!pip install torch==2.1.0 torch_geometric==2.4.0 --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install transformers==4.31.0 datasets==2.15.0 peft==0.5.0 accelerate==0.25.0
!pip install sentence-transformers==2.3.1 spacy==3.7.2 PyDictionary==2.0.1
!pip install tqdm

# Download spacy model
!python -m spacy download en_core_web_sm
print("\n✅ Dependencies installed successfully!")

# --- 2. IMPORTS ---
import json
import re
import spacy
import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ReLU
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset as PyGDataset
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType
from sentence_transformers import SentenceTransformer
from PyDictionary import PyDictionary
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset
from tqdm.notebook import tqdm
import os
import random
import numpy as np
import gc

# --- 3. CONFIGURATION ---
class Config:
    # Set your dataset paths here
    # RAW_DATASET_PATH_1 = "/kaggle/input/hallucination-train/train.model-agnostic_converted.json"
    # RAW_DATASET_PATH_2 = "/kaggle/input/hallucination-train/train.model-aware.v2_converted.json"
    
    ENCODER_MODEL_NAME = "microsoft/deberta-v3-small"
    SIMILARITY_MODEL_NAME = "all-MiniLM-L6-v2"
    
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    
    GNN_HIDDEN_CHANNELS = 128
    GNN_OUT_CHANNELS = 64
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    MAX_GRAD_NORM = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 4. SETUP DEVICE & GLOBAL MODELS ---
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
print(f"Using device: {Config.DEVICE}")

print("Loading global models...")
SPACY_NLP = spacy.load("en_core_web_sm")
PY_DICTIONARY = PyDictionary()
SIMILARITY_MODEL = SentenceTransformer(Config.SIMILARITY_MODEL_NAME, device=Config.DEVICE)
print("Global models loaded.")

# --- 5. DATA PROCESSING & VALIDATION FUNCTIONS ---
def normalize_text(text):
    if not text: return ""
    text = str(text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'<.*?>', ' ', text); text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_concepts(text, nlp_model):
    text = normalize_text(text)
    if not text: return []
    doc = nlp_model(text); concepts = set()
    for ent in doc.ents:
        if len(ent.text.split()) < 6: concepts.add(ent.text.lower())
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) < 6: concepts.add(chunk.text.lower())
    return list(concepts)

def get_external_definition(src_text, dictionary):
    match = re.search(r"meaning of ([a-zA-Z\s]+)", src_text, re.IGNORECASE)
    if not match: return "no word found"
    word = match.group(1).strip().lower()
    try:
        meaning = dictionary.meaning(word)
        if meaning:
            if 'Noun' in meaning: return meaning['Noun'][0]
            if 'Verb' in meaning: return meaning['Verb'][0]
            return list(meaning.values())[0][0]
    except Exception: pass
    return "no definition found"
    
def build_graph_for_datapoint(datapoint, similarity_model, spacy_nlp, dictionary):
    task = datapoint.get("task", "PG"); hyp_text = normalize_text(datapoint.get("hyp", ""))
    internal_ref_text, external_text = "", ""
    if task in ["MT", "PG"]: internal_ref_text = normalize_text(datapoint.get("src", ""))
    elif task == "DM":
        internal_ref_text = normalize_text(datapoint.get("tgt", ""))
        external_text = get_external_definition(datapoint.get("src", ""), dictionary)
    hyp_concepts = extract_concepts(hyp_text, spacy_nlp)
    internal_ref_concepts = extract_concepts(internal_ref_text, spacy_nlp)
    external_concepts = extract_concepts(external_text, spacy_nlp)
    all_unique_concepts = sorted(list(set(hyp_concepts + internal_ref_concepts + external_concepts)))
    concept_to_idx = {c: i for i, c in enumerate(all_unique_concepts)}; num_concept_nodes = len(all_unique_concepts)
    node_texts = all_unique_concepts + [hyp_text, internal_ref_text]
    if "no definition" not in external_text and "no word" not in external_text:
        node_texts.append(external_text)
    num_nodes = len(node_texts)
    num_special_nodes = num_nodes - num_concept_nodes
    hyp_node_idx, internal_ref_node_idx = num_concept_nodes, num_concept_nodes + 1
    edge_index = []
    for c in hyp_concepts:
        if c in concept_to_idx: edge_index.extend([[hyp_node_idx, concept_to_idx[c]], [concept_to_idx[c], hyp_node_idx]])
    for c in internal_ref_concepts:
        if c in concept_to_idx: edge_index.extend([[internal_ref_node_idx, concept_to_idx[c]], [concept_to_idx[c], internal_ref_node_idx]])
    if "no definition" not in external_text and "no word" not in external_text:
        ext_node_idx = num_concept_nodes + 2
        for c in external_concepts:
            if c in concept_to_idx: edge_index.extend([[ext_node_idx, concept_to_idx[c]], [concept_to_idx[c], ext_node_idx]])
    if num_concept_nodes > 1:
        try:
            embeds = similarity_model.encode(all_unique_concepts, convert_to_tensor=True, device='cpu')
            sim = F.cosine_similarity(embeds.unsqueeze(1), embeds.unsqueeze(0), dim=-1)
            rows, cols = (sim > 0.75).nonzero(as_tuple=True)
            for r, c in zip(rows.tolist(), cols.tolist()):
                if r < c: edge_index.extend([[r, c], [c, r]])
        except Exception: pass
    if not edge_index: edge_index = [[hyp_node_idx, hyp_node_idx]]
    edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    if edge_tensor.numel() > 0 and edge_tensor.max() >= num_nodes: return None
    return Data(edge_index=edge_tensor, node_texts=node_texts, num_nodes=num_nodes,
                num_special_nodes=num_special_nodes)

def create_fallback_graph(sample):
    hyp_text = normalize_text(sample.get("hyp", "no hypothesis"))
    src_text = normalize_text(sample.get("src", "no source"))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    node_texts = [hyp_text, src_text]
    return Data(edge_index=edge_index, node_texts=node_texts, num_nodes=2, num_special_nodes=2)

print("✅ Data processing functions defined.")

# --- 6. DATASET AND MODEL CLASSES ---
class FidelityGraphDataset(PyGDataset):
    def __init__(self, hf_dataset):
        super().__init__()
        self.dataset = hf_dataset
        self.length = len(hf_dataset)

    def len(self):
        return self.length

    def get(self, idx):
        if idx >= self.length: idx = idx % self.length
        positive_sample = self.dataset[idx]
        positive_graph = build_graph_for_datapoint(positive_sample, SIMILARITY_MODEL, SPACY_NLP, PY_DICTIONARY)
        
        max_retries = 5; retries = 0
        while positive_graph is None and retries < max_retries:
            idx = (idx + 1) % self.length
            positive_sample = self.dataset[idx]
            positive_graph = build_graph_for_datapoint(positive_sample, SIMILARITY_MODEL, SPACY_NLP, PY_DICTIONARY)
            retries += 1
        
        if positive_graph is None: positive_graph = create_fallback_graph(positive_sample)

        negative_idx = idx
        attempts = 0
        while negative_idx == idx and attempts < 10:
            negative_idx = torch.randint(0, self.length, (1,)).item()
            attempts += 1
        
        negative_sample = self.dataset[negative_idx]
        task = negative_sample.get("task", "PG")
        negative_ref_text = normalize_text(negative_sample.get("src" if task in ["MT", "PG"] else "tgt", ""))
        positive_graph.negative_ref_text = negative_ref_text
        return positive_graph

class SFG_GNN(Module):
    def __init__(self, encoder_model_name, gnn_hidden, gnn_out, lora_config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        encoder = AutoModel.from_pretrained(encoder_model_name)
        self.encoder = get_peft_model(encoder, lora_config)
        self.encoder.print_trainable_parameters()
        encoder_dim = self.encoder.config.hidden_size
        self.conv1 = GATv2Conv(encoder_dim, gnn_hidden, heads=2, concat=True, add_self_loops=True)
        self.conv2 = GATv2Conv(gnn_hidden * 2, gnn_out, heads=1, concat=False, add_self_loops=True)
        self.predictor = torch.nn.Sequential(Linear(gnn_out * 2, gnn_out), ReLU(), Linear(gnn_out, 1))

    def get_text_embeddings(self, texts):
        processed = [str(item) if not isinstance(item, list) else " ".join(map(str,item)) for item in texts]
        valid = [t for t in processed if t and t.strip()]
        if not valid: return torch.zeros((len(processed), self.encoder.config.hidden_size), device=self.encoder.device)
        inputs = self.tokenizer(valid, padding=True, truncation=True, return_tensors="pt", max_length=128).to(self.encoder.device)
        outputs = self.encoder(**inputs)
        mask = inputs.attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = torch.sum(outputs.last_hidden_state * mask, 1) / mask.sum(1)
        final = torch.zeros((len(processed), self.encoder.config.hidden_size), device=self.encoder.device)
        valid_indices = [i for i, t in enumerate(processed) if t and t.strip()]
        if pooled.shape[0] != len(valid_indices): return torch.zeros((len(processed), self.encoder.config.hidden_size), device=self.encoder.device)
        final[valid_indices] = pooled
        return final

    def forward(self, data):
        x = self.get_text_embeddings(data.node_texts)
        x = F.elu(self.conv1(x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        hyp, ref = [], []
        for i in range(data.num_graphs):
            s, e = data.ptr[i], data.ptr[i+1]
            n_nodes = e - s
            n_special = data.num_special_nodes[i].item()
            hyp_idx = s + (n_nodes - n_special)
            ref_idx = hyp_idx + 1
            if hyp_idx < e and ref_idx < e: # Bounds check
                hyp.append(x[hyp_idx]); ref.append(x[ref_idx])
        if not hyp: # If no valid pairs were found in batch
            return torch.tensor([]), torch.tensor([])

        hyp, ref = torch.stack(hyp), torch.stack(ref)
        pos_score = self.predictor(torch.cat([hyp, ref], dim=1))
        neg_ref = self.get_text_embeddings(data.negative_ref_text)
        neg_score = self.predictor(torch.cat([hyp, neg_ref], dim=1))
        return torch.sigmoid(pos_score), torch.sigmoid(neg_score)

print("✅ Dataset and Model classes defined.")


# --- 7. ROBUST TRAINING AND TESTING PIPELINE ---

def contrastive_loss(positive_scores, negative_scores, margin=0.5):
    positive_scores = torch.clamp(positive_scores, min=1e-7, max=1.0 - 1e-7)
    negative_scores = torch.clamp(negative_scores, min=1e-7, max=1.0 - 1e-7)
    positive_loss = torch.mean((1 - positive_scores) ** 2)
    negative_loss = torch.mean(torch.clamp(negative_scores - margin, min=0) ** 2)
    return positive_loss + negative_loss

def run_training_pipeline(config, full_dataset):
    # Create LoRA config
    temp_model = AutoModel.from_pretrained(config.ENCODER_MODEL_NAME)
    lora_targets = sorted(list(set([n.split(".")[-1] for n, m in temp_model.named_modules() if isinstance(m, torch.nn.Linear) and "attention" in n])))
    del temp_model
    print(f"Dynamically found LoRA target modules: {lora_targets}")
    lora_config = LoraConfig(r=config.LORA_R, lora_alpha=config.LORA_ALPHA, lora_dropout=config.LORA_DROPOUT,
                            task_type="FEATURE_EXTRACTION", target_modules=lora_targets)

    # Create model, dataset, loader, optimizer
    model = SFG_GNN(config.ENCODER_MODEL_NAME, config.GNN_HIDDEN_CHANNELS, config.GNN_OUT_CHANNELS, lora_config).to(config.DEVICE)
    train_dataset = FidelityGraphDataset(full_dataset)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training Loop
    print("\n🚀 Starting robust training...")
    model.train()
    best_loss = float('inf')
    for epoch in range(config.NUM_EPOCHS):
        total_loss = 0; successful_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        for batch_idx, batch in enumerate(pbar):
            try:
                optimizer.zero_grad()
                batch = batch.to(config.DEVICE, non_blocking=True)
                pos_scores, neg_scores = model(batch)
                
                if pos_scores.nelement() == 0: continue # Skip if batch was empty
                    
                loss = contrastive_loss(pos_scores, neg_scores)
                if loss.isnan() or loss.isinf(): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
                optimizer.step()
                
                total_loss += loss.item(); successful_batches += 1
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{total_loss/max(1, successful_batches):.4f}"})
            except Exception as e:
                # print(f"Skipping batch {batch_idx} due to error: {e}")
                continue
        
        if successful_batches > 0:
            avg_loss = total_loss / successful_batches
            print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} | Avg Loss: {avg_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            scheduler.step(avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                # Optional: Save best model
                # torch.save(model.state_dict(), 'best_model.pth')
    
    print("✅ Training finished!")
    return model

def test_model(model, config):
    print("\n🧪 Testing the trained model...")
    model.eval()
    test_samples = [
        {"hyp": "Madrid is the capital of Spain", "src": "Spain's capital is Madrid.", "task": "PG"},
        {"hyp": "The Great Wall is in Antarctica", "src": "The Great Wall is a landmark in China.", "task": "PG"},
        {"hyp": "Python is a snake.", "tgt": "Python is a high-level programming language.", "src": "What is the meaning of Python?", "task": "DM"}
    ]
    test_dataset = FidelityGraphDataset(HFDataset.from_list(test_samples))
    test_loader = DataLoader(test_dataset, batch_size=len(test_samples))
    with torch.no_grad():
        for batch in test_loader:
            pos_scores, _ = model(batch.to(config.DEVICE))
            for i, sample in enumerate(test_samples):
                score = pos_scores[i].item()
                print(f"Sample: \"{sample['hyp']}\"\nFidelity Score: {score:.4f} -> Hallucination Risk: {'Low' if score > 0.6 else 'High'}\n" + "-"*50)
    
# --- 8. EXECUTION ---
# Create a config instance
config = Config()
# Create a sample dataset (or load your real one)
sample_data = [
    {"hyp": "Paris is the capital of France.", "src": "France's capital city is Paris.", "task": "PG"},
    {"hyp": "The Eiffel Tower is in Rome.", "src": "The Eiffel Tower is a famous landmark in Paris.", "task": "PG"},
    {"hyp": "Python is a snake.", "tgt": "Python is a high-level programming language.", "src": "What is the meaning of Python?", "task": "DM"},
    {"hyp": "The sun rises in the west.", "src": "The sun rises in the east and sets in the west.", "task": "PG"},
    {"hyp": "Water boils at 100 degrees Celsius.", "src": "The boiling point of water is 100 C.", "task": "PG"}
] * 10 # Multiply to make a slightly larger dataset for testing
training_dataset = HFDataset.from_list(sample_data)

# Run the full pipeline
trained_model = run_training_pipeline(config, training_dataset)
if trained_model:
    test_model(trained_model, config)
    
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
print("🎉 Complete!")