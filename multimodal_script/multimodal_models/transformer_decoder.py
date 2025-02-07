import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerDecoder(nn.Module):
    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, embed_dim, num_heads, num_layers, num_classes=1, dropout=0.3, class_dropout=0.1):
        super().__init__()

        self.img_model = img_model  # Image feature extractor
        self.tab_model = tab_model  # Tabular data feature extractor
        self.embed_dim = embed_dim
        self.args = args

        # Project image features to a common embedding space
        self.img_proj = nn.Sequential(
            nn.Linear(img_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Project tabular features to a common embedding space
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Positional encoding for query (image features)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
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
            img_input: Tensor of image features, shape (batch_size, num_img_features, img_feature_dim)
            tab_input: Tensor of tabular features, shape (batch_size, num_tab_features, tab_feature_dim)
        Returns:
            logits: Classification logits for each sample in the batch, shape (batch_size, num_classes)
        """
        # Extract image features
        with torch.no_grad() if self.args.img_checkpoint else torch.enable_grad():
            img_features = self.img_model(img_input)  # (batch_size, img_feature_dim)
            
        img_features = self.img_proj(img_features).unsqueeze(0)  # Add sequence dimension: (1, batch_size, embed_dim)

        # Extract tabular features
        with torch.no_grad() if self.args.tab_checkpoint else torch.enable_grad():
            tab_features = self.tab_model(tab_input)  # (batch_size, tab_feature_dim)
            
        tab_features = self.tab_proj(tab_features).unsqueeze(0)  # Add sequence dimension: (1, batch_size, embed_dim)

        # Add positional encoding to image features (query)
        #img_features = self.positional_encoding(img_features)

        # Decode using cross-attention (image as query, tabular as key-value)
        decoded_features = self.decoder(
            tgt=img_features,  # Query features (images)
            memory=tab_features  # Key-value features (tabular data)
        )  

        # Remove sequence dimension and classify
        decoded_features = self.norm(decoded_features).squeeze(0)
        logits = self.classifier(decoded_features)  

        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

