import torch
from torch import nn

class TransformerPatientPerEncoder(nn.Module):
    def __init__(self, img_model, tab_model, img_feature_dim, tab_feature_dim, embed_dim, num_heads, num_layers, num_classes=1, dropout=0.1):
        super().__init__()
        
        self.img_model = img_model  # Modelo de imágenes
        self.tab_model = tab_model  # Modelo de información tabular
        
        # Fusion layer: Concatenar los features
        combined_dim = img_feature_dim + img_feature_dim
        
        # Projection layers for alignment
        self.tab_proj = nn.Linear(tab_feature_dim, img_feature_dim)  # Tab features remain as 64 (optional)
        
        self.embed_dim = embed_dim

        # Proyección de características de pacientes a embedding
        self.feature_embedding = nn.Linear(combined_dim, embed_dim)
        
        # Codificación posicional
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Clasificador para cada paciente
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 512),  # Primera capa con más unidades
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),  # Reduciendo gradualmente la dimensionalidad
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)  # Salida binaria
        )
        
    def forward(self, img_input, tab_input):
        """
        Args:
            x: Tensor de entrada de forma (batch_size, num_patients, num_features)
        Returns:
            logits: Tensor de salida de forma (batch_size, num_patients, num_classes)
        """
        #breakpoint()
        with torch.no_grad():
            img_features = self.img_model(img_input)
        
        # Obtener los features de la información tabular
        with torch.no_grad():
            tab_features = self.tab_model(tab_input)
        tab_features = self.tab_proj(tab_features)
        
        # Concatenar ambos features
        combined_features = torch.cat((img_features, tab_features), dim=1)
        
        combined_features = combined_features.unsqueeze(0)
        # Proyectar las características de pacientes a embeddings
        x = self.feature_embedding(combined_features)  # (batch_size, num_patients, embed_dim)

        # Añadir codificación posicional
        x = self.positional_encoding(x)  # (batch_size, num_patients, embed_dim)

        # Pasar por el Transformer Encoder
        x = self.encoder(x)  # (batch_size, num_patients, embed_dim)

        # Clasificar cada paciente
        logits = self.classifier(x)  # (batch_size, num_patients, num_classes)
        return logits.squeeze(0)


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
