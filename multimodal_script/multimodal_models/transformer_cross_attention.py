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

        # Transformer Decoder
        """
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            batch_first=True
        )
        """
        decoder_layer = CustomTransformerDecoderLayer(
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
        
    def forward(self, img_input, tab_input, return_attn=False):
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
        
        if return_attn:
            self_attn_img = self.decoder_img.layers[0].self_attn_weights  # Self-attention for images
            cross_attn_img = self.decoder_img.layers[0].cross_attn_weights  # Cross-attention for images
            self_attn_tab = self.decoder_tab.layers[0].self_attn_weights  # Self-attention for tabular
            cross_attn_tab = self.decoder_tab.layers[0].cross_attn_weights  # Cross-attention for tabular
            return logits, self_attn_img, cross_attn_img, self_attn_tab, cross_attn_tab
            
        return logits

# Modified Decoder to store attention weigths for visualization
import torch.nn as nn
import torch

class CustomTransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.self_attn_weights = None  # Store self-attention weights
        self.cross_attn_weights = None  # Store cross-attention weights

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, 
                tgt_is_causal=None, memory_is_causal=None):
        """
        Handles self-attention and cross-attention storage.
        - tgt: Query sequence (modality itself, e.g., image or tabular features)
        - memory: Key-Value sequence (other modality)
        """

        # Self-Attention: How much tgt attends to itself
        tgt2, self_attn_weights = self.self_attn(
            tgt, tgt, tgt,  
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=True  
        )
        self.self_attn_weights = self_attn_weights

        # Cross-Attention: How much tgt attends to memory
        tgt3, cross_attn_weights = self.multihead_attn(
            tgt2, memory, memory,  
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True  
        )
        self.cross_attn_weights = cross_attn_weights

        # Continue normal processing with correct arguments
        return super().forward(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal
        )
