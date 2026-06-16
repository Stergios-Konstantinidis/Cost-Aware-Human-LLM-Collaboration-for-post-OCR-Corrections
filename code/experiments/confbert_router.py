"""
confbert_router.py
ConfBERT-style OCR error detection model for routing.

Implements the confidence-aware embedding injection from Hemmer et al.:
    ec_i = (1 - α) * Emb(t_i) + α * (1 - p_ocr(t_i))
where α is a learnable parameter and p_ocr is the OCR confidence.

The model is fine-tuned on our historical French newspaper data for
binary classification: does this document benefit from LLM correction?

Usage:
    from confbert_router import train_confbert_router
    probas = train_confbert_router(records, corrections, results_dir)
"""
import json
import warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from transformers import AutoTokenizer, AutoModel


# ── ConfBERT Model ───────────────────────────────────────────────────────────

class ConfBERTClassifier(nn.Module):
    """
    BERT with confidence-aware embedding injection.
    
    For each token t_i with OCR confidence p_ocr(t_i):
        ec_i = (1 - α) * Emb(t_i) + α * (1 - p_ocr(t_i))
    
    α is a learnable scalar parameter.
    The [CLS] representation is then used for document-level classification.
    """
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased",
                 num_labels: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        
        # Learnable mixing parameter α (initialized small)
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        # Projection from scalar confidence to embedding space
        self.conf_projection = nn.Linear(1, hidden_size)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_labels)
        )
    
    def forward(self, input_ids, attention_mask, token_confidences):
        """
        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            token_confidences: (B, L) per-token OCR confidence in [0, 1]
        Returns:
            logits: (B, 2) classification logits
        """
        # Get original BERT embeddings (before transformer layers)
        embeddings = self.bert.embeddings(input_ids)
        
        # Confidence injection: ec = (1-α)*emb + α*proj(1 - conf)
        alpha = torch.sigmoid(self.alpha)  # constrain to [0, 1]
        conf_signal = (1.0 - token_confidences).unsqueeze(-1)  # (B, L, 1)
        conf_emb = self.conf_projection(conf_signal)            # (B, L, H)
        
        modified_embeddings = (1.0 - alpha) * embeddings + alpha * conf_emb
        
        # Pass through BERT encoder (using modified embeddings)
        extended_mask = self.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )
        encoder_output = self.bert.encoder(
            modified_embeddings,
            attention_mask=extended_mask
        )
        
        # Use [CLS] token for classification
        cls_output = encoder_output.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


# ── Dataset ──────────────────────────────────────────────────────────────────

class OCRDocumentDataset(Dataset):
    """Dataset of OCR documents with per-token confidence scores."""
    
    def __init__(self, texts: List[str], labels: np.ndarray,
                 token_confidences: List[Dict[str, float]],
                 tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.token_confidences = token_confidences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        word_confs = self.token_confidences[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offsets = encoding['offset_mapping'].squeeze(0)
        
        # Map word-level confidences to subword tokens
        token_conf = self._align_confidences(text, word_confs, offsets, attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_confidences': token_conf,
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _align_confidences(self, text, word_confs, offsets, attention_mask):
        """Map word-level OCR confidences to subword token positions."""
        words = text.split()
        conf_values = []
        for w in words:
            conf_values.append(word_confs.get(w.lower(), word_confs.get('_avg', 0.95)))
        
        # Build character-to-confidence mapping
        char_conf = np.ones(len(text), dtype=np.float32) * word_confs.get('_avg', 0.95)
        pos = 0
        for i, word in enumerate(words):
            start = text.find(word, pos)
            if start >= 0:
                for j in range(start, start + len(word)):
                    if j < len(char_conf):
                        char_conf[j] = conf_values[i] if i < len(conf_values) else 0.95
                pos = start + len(word)
        
        # Map to subword tokens via offset_mapping
        token_conf = torch.ones(len(offsets), dtype=torch.float32) * word_confs.get('_avg', 0.95)
        for i, (start, end) in enumerate(offsets):
            if start == 0 and end == 0:
                # Special token [CLS], [SEP], [PAD]
                token_conf[i] = 1.0
            elif start < len(char_conf):
                # Average confidence over the character span
                token_conf[i] = float(np.mean(char_conf[start:max(end, start+1)]))
        
        return token_conf


# ── Token-Level Confidence Extraction ────────────────────────────────────────

def extract_word_confidences(img_path: Path) -> Dict[str, float]:
    """Extract per-word OCR confidence from Tesseract for a single image."""
    import pytesseract
    import pandas as pd
    from io import StringIO
    
    try:
        data = pytesseract.image_to_data(str(img_path), lang='fra')
        df = pd.read_csv(StringIO(data), sep='\t', quoting=3)
        df = df[df['conf'] != -1]
        df = df[df['text'].apply(lambda x: str(x).strip() != '')]
        
        word_confs = {}
        for _, row in df.iterrows():
            word = str(row['text']).strip().lower()
            conf = float(row['conf']) / 100.0  # normalize to [0, 1]
            if word:
                word_confs[word] = conf
        
        avg_conf = df['conf'].mean() / 100.0 if len(df) > 0 else 0.95
        word_confs['_avg'] = avg_conf
        return word_confs
    except Exception:
        return {'_avg': 0.95}


def load_or_compute_word_confidences(records, results_dir, images_dir) -> List[Dict[str, float]]:
    """Load cached word-level confidences or compute from images."""
    cache_path = Path(results_dir) / "confidence_data/word_confidences_tesseract.json"
    
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"  Loaded cached word confidences for {len(cached)} documents")
    else:
        cached = {}
    
    all_confs = []
    to_compute = []
    
    for r in records:
        fn = r['filename']
        if fn in cached:
            all_confs.append(cached[fn])
        else:
            to_compute.append((len(all_confs), fn))
            # Use document-level confidence as placeholder
            avg_conf = float(r.get('_avg_confidence', 0.95))
            all_confs.append({'_avg': avg_conf})
    
    if to_compute:
        print(f"  Computing word confidences for {len(to_compute)} new documents...")
        from tqdm import tqdm
        img_dir = Path(images_dir)
        for idx, fn in tqdm(to_compute, desc="Extracting word conf"):
            img_path = img_dir / fn
            if img_path.exists():
                wc = extract_word_confidences(img_path)
                all_confs[idx] = wc
                cached[fn] = wc
            else:
                # Use _avg_confidence from document-level
                avg_conf = float(records[idx].get('_avg_confidence', 0.95))
                all_confs[idx] = {'_avg': avg_conf}
                cached[fn] = all_confs[idx]
        
        # Save cache
        with open(cache_path, 'w') as f:
            json.dump(cached, f)
        print(f"  Saved word confidences to {cache_path}")
    
    return all_confs


# ── Training ─────────────────────────────────────────────────────────────────

def train_confbert_router(
    records: List[dict],
    corrections: Dict[str, dict],
    results_dir: str,
    images_dir: str = "data/evaluation_dataset/images",
    metric: str = "cer",
    min_delta: float = 0.0,
    n_splits: int = 10,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    max_length: int = 256,
    model_name: str = "bert-base-multilingual-cased",
) -> np.ndarray:
    """
    Train ConfBERT via cross-validation and return out-of-fold routing probabilities.
    
    Returns:
        probas: (N,) array of routing probabilities (higher = more likely to need correction)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  ConfBERT device: {device}")
    
    # Prepare texts and labels
    texts = [r['raw_ocr'] for r in records]
    base_metric = np.array([float(r[metric]) for r in records])
    corr_metric = np.array([
        corrections.get(r['filename'], {}).get(metric, float(r[metric]))
        for r in records
    ])
    deltas = base_metric - corr_metric
    labels = (deltas > min_delta).astype(int)
    
    print(f"  Labels: {labels.sum()} positive / {len(labels) - labels.sum()} negative")
    
    # Load word-level confidences
    word_confs = load_or_compute_word_confidences(records, results_dir, images_dir)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_probas = np.zeros(len(records), dtype=np.float32)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(records)):
        print(f"  Fold {fold+1}/{n_splits}...")
        
        # Create datasets
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        train_confs = [word_confs[i] for i in train_idx]
        val_confs = [word_confs[i] for i in val_idx]
        
        train_ds = OCRDocumentDataset(train_texts, train_labels, train_confs,
                                       tokenizer, max_length)
        val_ds = OCRDocumentDataset(val_texts, val_labels, val_confs,
                                     tokenizer, max_length)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = ConfBERTClassifier(model_name=model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Handle class imbalance
        pos_weight = (len(train_labels) - train_labels.sum()) / max(train_labels.sum(), 1)
        weights = torch.tensor([1.0, pos_weight], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        
        # Training loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_conf = batch['token_confidences'].to(device)
                batch_labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                logits = model(input_ids, attention_mask, token_conf)
                loss = criterion(logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
        
        # Validation predictions
        model.eval()
        fold_probas = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_conf = batch['token_confidences'].to(device)
                
                logits = model(input_ids, attention_mask, token_conf)
                probs = torch.softmax(logits, dim=1)[:, 1]
                fold_probas.extend(probs.cpu().numpy())
        
        oof_probas[val_idx] = np.array(fold_probas)
        
        # Clean up GPU memory
        del model, optimizer
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    print(f"  ConfBERT OOF probas: mean={oof_probas.mean():.4f}, std={oof_probas.std():.4f}")
    return oof_probas
