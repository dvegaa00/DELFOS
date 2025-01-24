import torch
from torch import nn
import pathlib 

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from multimodal_script.multimodal_models.mlp import MLP
from multimodal_script.multimodal_models.transformer_encoder import TransformerPatientPerEncoder

class MultimodalModel:
    def __init__(self, img_model, tab_model, args):
        """
        Initialize the ModelFactory with the given arguments.

        Args:
            args (Namespace): Parsed arguments containing model configuration.
        """
        self.img_model = img_model
        self.tab_model = tab_model
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        """
        Builds and initializes the model based on the provided arguments.

        Returns:
            nn.Module: The constructed model.
        """
        if self.args.multimodal_model == "MPL":
            self.model = MLP(img_model=self.img_model,
                            tab_model=self.tab_model,
                            img_feature_dim=self.args.img_feature_dim,
                            tab_feature_dim=self.args.tab_feature_dim)
            
        elif self.args.multimodal_model == "TransEnc":
            self.model = TransformerPatientPerEncoder(
                                img_model=self.img_model,
                                tab_model=self.tab_model,
                                img_feature_dim=self.args.img_feature_dim,
                                tab_feature_dim=self.args.tab_feature_dim,
                                embed_dim=self.args.embed_dim,
                                num_heads=self.args.num_heads,
                                num_layers=self.args.num_layers,
                                num_classes=self.args.n_classes  # Clasificación binaria
                            )
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        return self.model
    
    
    
