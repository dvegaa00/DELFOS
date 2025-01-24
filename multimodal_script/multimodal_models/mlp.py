import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, img_model, tab_model, img_feature_dim, tab_feature_dim):
        super(MLP, self).__init__()
        self.img_model = img_model  # Modelo de imágenes
        self.tab_model = tab_model  # Modelo de información tabular
        
        # Fusion layer: Concatenar los features
        combined_dim = img_feature_dim + img_feature_dim
        
        # Projection layers for alignment
        self.tab_proj = nn.Linear(tab_feature_dim, img_feature_dim)  # Tab features remain as 64 (optional)
        
        # MLP
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),  # Primera capa con más unidades
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),  # Reduciendo gradualmente la dimensionalidad
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # Salida binaria
        )

    def forward(self, img_input, tab_input):
        # Obtener los features de las imágenes
        #breakpoint()
        with torch.no_grad():
            img_features = self.img_model(img_input)
        
        # Obtener los features de la información tabular
        with torch.no_grad():
            tab_features = self.tab_model(tab_input)
        tab_features = self.tab_proj(tab_features)
        
        # Concatenar ambos features
        combined_features = torch.cat((img_features, tab_features), dim=1)
        
        # Pasar por el clasificador
        output = self.classifier(combined_features)
        return output

