import torch
from torch import nn
import pathlib 
import sys 

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from multimodal_script.multimodal_models.mlp import MLP
from multimodal_script.multimodal_models.transformer_encoder import TransformerPatientPerEncoder
from multimodal_script.multimodal_models.transformer_decoder import TransformerDecoder
from multimodal_script.multimodal_models.transformer_self_cross import TransformerSelfCross
from multimodal_script.multimodal_models.transformer_cross_attention import TransformerCrossAttention
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
        if self.args.multimodal_model == "mlp":
            self.model = MLP(args = self.args,
                             img_model=self.img_model,
                             tab_model=self.tab_model,
                             img_feature_dim=self.args.img_feature_dim,
                             tab_feature_dim=self.args.tab_feature_dim,
                             class_dropout=self.args.class_dropout
                            )
            
        elif self.args.multimodal_model == "TransEncoder":
            self.model = TransformerPatientPerEncoder(
                                args = self.args,
                                img_model=self.img_model,
                                tab_model=self.tab_model,
                                img_feature_dim=self.args.img_feature_dim,
                                tab_feature_dim=self.args.tab_feature_dim,
                                embed_dim=self.args.embed_dim,
                                num_heads=self.args.num_heads,
                                num_layers=self.args.num_layers,
                                num_classes=self.args.n_classes,  # Clasificación binaria
                                dropout=self.args.path_dropout,
                                class_dropout=self.args.class_dropout
                            )
            
        elif self.args.multimodal_model == "TransDecoder":
            self.model = TransformerDecoder(
                                args = self.args,
                                img_model=self.img_model,
                                tab_model=self.tab_model,
                                img_feature_dim=self.args.img_feature_dim,
                                tab_feature_dim=self.args.tab_feature_dim,
                                embed_dim=self.args.embed_dim,
                                num_heads=self.args.num_heads,
                                num_layers=self.args.num_layers,
                                num_classes=self.args.n_classes,  # Clasificación binaria
                                dropout=self.args.path_dropout,
                                class_dropout=self.args.class_dropout
                            )
            
        elif self.args.multimodal_model == "TransCross":
            self.model = TransformerSelfCross(
                                args = self.args,
                                img_model=self.img_model,
                                tab_model=self.tab_model,
                                img_feature_dim=self.args.img_feature_dim,
                                tab_feature_dim=self.args.tab_feature_dim,
                                embed_dim=self.args.embed_dim,
                                num_heads=self.args.num_heads,
                                num_layers=self.args.num_layers,
                                num_classes=self.args.n_classes,  # Clasificación binaria
                                dropout=self.args.path_dropout,
                                class_dropout=self.args.class_dropout
                            )
            
        elif self.args.multimodal_model == "TransDoubleCross":
            self.model = TransformerCrossAttention(
                                args = self.args,
                                img_model=self.img_model,
                                tab_model=self.tab_model,
                                img_feature_dim=self.args.img_feature_dim,
                                tab_feature_dim=self.args.tab_feature_dim,
                                embed_dim=self.args.embed_dim,
                                num_heads=self.args.num_heads,
                                num_layers=self.args.num_layers,
                                num_classes=self.args.n_classes,  # Clasificación binaria
                                dropout=self.args.path_dropout,
                                class_dropout=self.args.class_dropout                          
                            )
            
        else:
            raise ValueError(f"Unsupported model: {self.args.multimodal_model}")

        return self.model
    
    
    
