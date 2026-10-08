"""
FT-Transformer / PyTorch Tabular Model for Procurement Corruption Risk Classification.
Feature Tokenizer Transformer architecture for structured tabular and text risk features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


class TabularFTTransformer(nn.Module):
    """
    Feature Tokenizer + Transformer (FT-Transformer) Architecture for Tabular Data.
    Converts numerical/categorical features into feature tokens and applies self-attention.
    """
    def __init__(self, num_features: int, embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2, num_classes: int = 3):
        super(TabularFTTransformer, self).__init__()
        
        # Feature Tokenizer: Linear projections per feature
        self.feature_encoders = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])
        
        # Transformer Encoder Stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size = x.shape[0]
        
        # Tokenize each feature
        tokens = []
        for i, encoder in enumerate(self.feature_encoders):
            feature_val = x[:, i:i+1] # (batch_size, 1)
            token = encoder(feature_val) # (batch_size, embed_dim)
            tokens.append(token.unsqueeze(1)) # (batch_size, 1, embed_dim)
            
        x_tokens = torch.cat(tokens, dim=1) # (batch_size, num_features, embed_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_all = torch.cat([cls_tokens, x_tokens], dim=1) # (batch_size, num_features + 1, embed_dim)
        
        # Pass through Transformer
        transformed = self.transformer(x_all)
        
        # Take CLS token embedding output
        cls_out = transformed[:, 0, :]
        logits = self.classifier(cls_out)
        return logits


class ProcurementDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_ft_transformer(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, num_classes: int = 3, epochs: int = 15, batch_size: int = 32):
    """
    Trains FT-Transformer model on tabular risk features.
    """
    train_dataset = ProcurementDataset(X_train, y_train)
    val_dataset = ProcurementDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    num_features = X_train.shape[1]
    model = TabularFTTransformer(num_features=num_features, num_classes=num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)
            
        train_loss = total_loss / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y_batch).sum().item()
                
        val_acc = correct / len(val_dataset)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss/len(val_dataset):.4f} | Val Acc: {val_acc:.4f}")
            
    return model
