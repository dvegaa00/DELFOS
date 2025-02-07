import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, args, img_model, tab_model, img_feature_dim, tab_feature_dim, class_dropout=0.1):
        super(MLP, self).__init__()
        self.img_model = img_model  # Modelo de imágenes
        self.tab_model = tab_model  # Modelo de información tabular
        self.args = args
        self.num_classes = args.n_classes
        # Fusion layer: Concatenar los features
        combined_dim = img_feature_dim + img_feature_dim
        
        # Projection layers for alignment
        self.img_proj = nn.Sequential(nn.Linear(img_feature_dim, img_feature_dim),
                                      nn.LayerNorm(img_feature_dim),
                                      nn.ReLU(inplace=True))
                                      
        self.tab_proj = nn.Sequential(nn.Linear(tab_feature_dim, img_feature_dim),
                                      nn.LayerNorm(img_feature_dim),
                                      nn.ReLU(inplace=True))
        
        self.norm = nn.LayerNorm(combined_dim)
        
        # MLP
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, combined_dim//2),  # Primera capa con más unidades
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(combined_dim//2, combined_dim//4),  # Reduciendo gradualmente la dimensionalidad
            nn.ReLU(),
            nn.Dropout(class_dropout),
            nn.Linear(combined_dim//4, self.num_classes)  # Salida binaria
        )

    def forward(self, img_input, tab_input):
        # Obtener los features de las imágenes
        if self.args.img_checkpoint != None:
            with torch.no_grad():
                img_features = self.img_model(img_input)
        else:
            img_features = self.img_model(img_input)
        img_features = self.img_proj(img_features)
            
        # Obtener los features de la información tabular
        if self.args.tab_checkpoint != None:
            with torch.no_grad():
                tab_features = self.tab_model(tab_input)
        else:
            tab_features = self.tab_model(tab_input)
        tab_features = self.tab_proj(tab_features)
        
        # Concatenar ambos features
        combined_features = torch.cat((img_features, tab_features), dim=1)
        combined_features = self.norm(combined_features)
        
        # Pasar por el clasificador
        output = self.classifier(combined_features)
        return output

