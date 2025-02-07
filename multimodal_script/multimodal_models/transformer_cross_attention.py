import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerCrossAttention(nn.Module):
    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
        super().__init__()
        
        self.img_model = img_model  # Image model
        self.tab_model = tab_model  # Tabular model
        self.embed_dim = embed_dim
        self.args = args
        
        # Projection layers for alignment
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Positional encoding for image features (query)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        
        self.decoder_img = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.decoder_tab = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.final_norm_img = nn.LayerNorm(embed_dim)
        self.final_norm_tab = nn.LayerNorm(embed_dim)

        # Classifier for every patient
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(embed_dim // 4, num_classes)
        )
        
    def forward(self, img_input, tab_input):
        """
        Args:
            img_input: Image input tensor (batch_size, num_img_features, img_feature_dim)
            tab_input: Tabular input tensor (batch_size, num_tab_features, tab_feature_dim)
        Returns:
            logits: Output logits for classification (batch_size, num_classes)
        """
        # Extract features from the image model
        with torch.no_grad() if self.args.img_checkpoint else torch.enable_grad():
            img_features = self.img_model(img_input)
        img_features = self.img_proj(img_features).unsqueeze(0)
        
        # Extract features from the tabular model
        with torch.no_grad() if self.args.tab_checkpoint else torch.enable_grad():
            tab_features = self.tab_model(tab_input)
        tab_features = self.tab_proj(tab_features).unsqueeze(0)
        
        # Add positional encoding to image and tabular features
        #img_features = self.positional_encoding(img_features)
        #tab_features = self.positional_encoding(tab_features)
        
        img_features = self.decoder_img(
            tgt=img_features,  # Query features (images)
            memory=tab_features  # Key-value features (tabular data)
        )  
        
        tab_features = self.decoder_tab(
            tgt=tab_features,  # Query features (tabular data)
            memory=img_features  # Key-value features (images)
        )  

        # Classify each patient
        output_img = self.final_norm_img(img_features).squeeze(0)
        output_tab = self.final_norm_tab(tab_features).squeeze(0)
        
        output = torch.cat((output_img, output_tab), dim=1)
        
        logits = self.classifier(output)
        
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Generar codificación posicional
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Cross-Attention Layer
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, key_value):
        attn_output, _ = self.multihead_attn(query, key_value, key_value)
        return self.norm(query + self.dropout(attn_output))
