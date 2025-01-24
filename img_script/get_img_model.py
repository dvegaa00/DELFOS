import torch
from torch import nn
from timm import create_model
import pathlib 

delfos_path = pathlib.Path(__name__).resolve().parent.parent
sys.path.append(str(delfos_path))
from DELFOS.img_script.img_models.MedViT import MedViT_small

class ImageModel:
    def __init__(self, args):
        """
        Initialize the ModelFactory with the given arguments.

        Args:
            args (Namespace): Parsed arguments containing model configuration.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def build_model(self):
        """
        Builds and initializes the model based on the provided arguments.

        Returns:
            nn.Module: The constructed model.
        """
        if self.args.img_model == "resnet18":
            pass
        
        elif self.args.img_model == "resnet50":
            pass
        
        elif self.args.img_model == "vit_tiny":
            self.model = create_model("vit_tiny_patch16_224", pretrained=self.args.img_pretrained, num_classes=self.args.n_classes).to(self.device)

        elif self.args.img_model == "vit_small":
            self.model = create_model("vit_small_patch16_224", pretrained=self.args.img_pretrained, num_classes=self.args.n_classes).to(self.device)

        elif self.args.img_model == "medvit":
            model = MedViT_small(num_classes=self.args.n_classes, path_dropout=0.1, attn_drop=0.3, drop=0.3).to(self.device)
            if self.args.img_pretrain:
                self.model = self.load_medvit_pretrained(model)
            else:
                print("No pretrained weights uploaded.")
                self.model = model
        else:
            raise ValueError(f"Unsupported model: {self.args.model}")

        return self.model

    def load_medvit_pretrained(self, model):
        """Load pretrained weights to MedViT Small.

        Args:
            model: MedViT model instance.

        Returns:
            nn.Module: Model with uploaded weights.
        """
        checkpoint_path = "/home/dvegaa/DELFOS/MedViT/MedViT_small_im1k.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        else:
            checkpoint_model = checkpoint

        # Filter out `proj_head` and load the rest
        filtered_checkpoint = {k: v for k, v in checkpoint_model.items() if not k.startswith("proj_head")}
        model.load_state_dict(filtered_checkpoint, strict=False)
        print("Checkpoint loaded successfully (excluding proj_head).")

        # Replace `proj_head` for the specific number of classes
        model.proj_head = nn.Sequential(nn.Linear(in_features=1024, out_features=self.args.n_classes, bias=True)).to(self.device)
        return model

        


    
    
    
