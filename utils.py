import argparse

str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description="Training configuration for Vision Transformer")

    # Add arguments
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.000001, help="Learning rate for optimizer")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Embedding dimension")
    parser.add_argument("--loss_factor", type=float, default=0.5, help="factor to multiply weight_loss")
    parser.add_argument("--img_pretrain", type=str2bool, default=True, help="True for pretrained image model")
    parser.add_argument("--img_model", type=str, default="medvit", help="whether to use medvit, vit_tiny, vity_small, resnet18, resnet50")
    parser.add_argument("--tab_pretrain", type=str2bool, default=True, help="True for pretrained tabular model")
    parser.add_argument("--tab_model", type=str, default="TabTransformer", help="whether to use TabTransformer")
    parser.add_argument("--multimodal_pretrain", type=str2bool, default=True, help="True for pretrained multimodal model")
    parser.add_argument("--multimodal_model", type=str, default="TransEncoder", help="whether to use MLP, Transformer Encoder, Transformer Decoder")
    parser.add_argument("--img_feature_dim", type=int, default=1024, help="Dimension of image features to be used in multimodal model")
    parser.add_argument("--tab_feature_dim", type=int, default=64, help="Dimension of tabular features to be used in multimodal model")
    parser.add_argument("--img_checkpoint", type=str, default=None, help="Path to image model checkpoint")
    parser.add_argument("--tab_checkpoint", type=str, default=None, help="Path to tabular model checkpoint")
    parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dimensions of the TransformerEncoder model")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of heads of the TransformerEncoder model")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers of the TransformerEncoder model")
    
    return parser.parse_args()


