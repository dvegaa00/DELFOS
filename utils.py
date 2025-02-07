import argparse
import io
import os
import time
from collections import defaultdict, deque
import datetime
import torch
import torch.distributed as dist
from torch import nn, einsum
from PIL import Image
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, precision_recall_curve, roc_auc_score, classification_report
from tqdm import tqdm
import torch
from tqdm import tqdm
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

str2bool = lambda x: (str(x).lower() == 'true')

def get_main_parser():
    parser = argparse.ArgumentParser(description="Training configuration for Vision Transformer")

    # Add arguments
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.000001, help="Learning rate for optimizer")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output classes")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Embedding dimension")
    parser.add_argument("--loss_factor", type=float, default=2, help="factor to multiply weight_loss")
    parser.add_argument("--img_pretrain", type=str2bool, default=False, help="True for pretrained image model")
    parser.add_argument("--img_model", type=str, default="vit_small", help="whether to use medvit, vit_tiny, vity_small, resnet18, resnet50")
    parser.add_argument("--tab_pretrain", type=str2bool, default=False, help="True for pretrained tabular model")
    parser.add_argument("--tab_model", type=str, default="TabTransformer", help="whether to use TabTransformer")
    parser.add_argument("--multimodal_pretrain", type=str2bool, default=False, help="True for pretrained multimodal model")
    parser.add_argument("--multimodal_model", type=str, default="TransDoubleCross", help="whether to use MLP, Transformer Encoder, Transformer Decoder")
    parser.add_argument("--img_feature_dim", type=int, default=384, help="Dimension of image features to be used in multimodal model")
    parser.add_argument("--tab_feature_dim", type=int, default=64, help="Dimension of tabular features to be used in multimodal model")
    parser.add_argument("--img_checkpoint", type=str, default="vit_small_0.3_0.1", help="Path to image model checkpoint")
    parser.add_argument("--tab_checkpoint", type=str, default="/home/dvegaa/DELFOS/DELFOS/tabular_script/tabular_checkpoints/model_2025-01-20_13-42-37.pth", help="Path to tabular model checkpoint")
    parser.add_argument("--embed_dim", type=int, default=384, help="Embedding dimensions of the TransformerEncoder model")
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads of the TransformerEncoder model")
    parser.add_argument("--num_layers", type=int, default=8, help="Number of layers of the TransformerEncoder model")
    parser.add_argument("--folds", type=int, default=3, help="Number of folds in k-fold cross validation")
    parser.add_argument("--path_dropout", type=float, default=0.3, help="Path dropout to be used in multimodal models")
    parser.add_argument("--class_dropout", type=float, default=0.1, help="Path dropout to be used in multimodal models")
    parser.add_argument("--exp_name", type=str, default="", help="Experiment name for optimization threshold")
    parser.add_argument("--sampling", type=str2bool, default=True, help="Whether to use weighted random sampling in dataloader or not")
    parser.add_argument("--smooth_label", type=str2bool, default=False, help="Whether to use smooth label or not")

    return parser.parse_args()

def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    
def plot_prcurve(pr_curve, path):
    # Plot PR curves for train and test
    plt.figure(figsize=(12, 6))

    # Test PR curve subplot
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    for i, (precision, recall) in enumerate(zip(pr_curve["precision_test"], pr_curve["recall_test"]), 1):
        plt.plot(recall, precision, label=f'Fold {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Test Precision-Recall Curve')
    plt.legend()
    plt.grid()

    # Train PR curve subplot
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    for i, (precision, recall) in enumerate(zip(pr_curve["precision_train"], pr_curve["recall_train"]), 1):
        plt.plot(recall, precision, label=f'Fold {i}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Train Precision-Recall Curve')
    plt.legend()
    plt.grid()

    # Save figure
    plt.tight_layout()
    plt.savefig(path)
    
    
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def get_best_threshold(y_true, y_score, num_thresholds=100):
    """
    Encuentra el mejor threshold basado en el F1-score generando manualmente los umbrales.

    Args:
        y_true (np.array): Etiquetas verdaderas (0 o 1).
        y_score (np.array): Puntajes o probabilidades del modelo.
        num_thresholds (int): Número de umbrales a probar (default=100).

    Returns:
        best_threshold (float): Mejor threshold basado en el F1-score.
        precision_list (np.array): Lista de valores de precisión.
        recall_list (np.array): Lista de valores de recall.
        thresholds (np.array): Lista de thresholds usados.
    """
    # Generar manualmente los thresholds en el rango [0,1]
    thresholds = np.linspace(0, 1, num_thresholds)

    precision_list = []
    recall_list = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=1)
        f1 = f1_score(y_true, y_pred, zero_division=1)

        precision_list.append(precision)
        recall_list.append(recall)
        f1_scores.append(f1)

    # Encontrar el índice del mejor F1-score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    return best_threshold, np.array(precision_list), np.array(recall_list), thresholds


def evaluate_threshold_img(model, loader, device, threshold=0.5):
    """
    Evaluate the model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        threshold (float): Threshold for binary classification.

    Returns:
        tuple: Precision, recall, accuracy, F1-score, predicted scores (y_score), and true labels (y_true).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            img_data, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.unsqueeze(1).to(device)

            # Forward pass
            outputs = model(img_data)
            targets = targets.to(torch.float32)

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())
    y_pred = [1 if score > threshold else 0 for score in y_score]  # Apply threshold of 0.5

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["No Cardiopatia", "Cardiopatia"], output_dict=True)

    return precision, recall, accuracy, f1, roc_auc, report, y_score, y_true

def find_best_threshold_img(model, loader, device):
    """
    Find the best threshold based on the F1-score using the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Best threshold for classification based on F1-score.
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Finding Best Threshold"):
            img_data, patient_ids = inputs[0].to(device), inputs[1]
            targets = targets.to(device)
            
            outputs = model(img_data)
            outputs = outputs.squeeze(1)
            
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())

    # Get best threshold
    best_threshold, precision, recall, thresholds = get_best_threshold(y_true, y_score)

    return best_threshold, precision, recall, thresholds


def evaluate_threshold_multimodal(model, loader, device, threshold=0.5):
    """
    Evaluate the model on the given data loader and calculate metrics.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').
        threshold (float): Threshold for binary classification.

    Returns:
        tuple: Precision, recall, accuracy, F1-score, predicted scores (y_score), and true labels (y_true).
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating"):
            img_data, tab_data, patient_ids = inputs[0].to(device), inputs[1].to(device), inputs[2]
            targets = targets.to(device)

            # Forward pass
            outputs = model(img_data, tab_data).squeeze(1)
            targets = targets.to(torch.float32)

            # Sigmoid to convert logits to probabilities
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()

            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target  

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())
    y_pred = [1 if score > threshold else 0 for score in y_score]  # Apply threshold of 0.5

    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)

    return precision, recall, accuracy, f1, roc_auc, y_score, y_true

def find_best_threshold_multimodal(model, loader, device):
    """
    Find the best threshold based on the F1-score using the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Best threshold for classification based on F1-score.
    """
    y_true_patient = {}
    y_score_patient = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Finding Best Threshold"):
            img_data, tab_data, patient_ids = inputs[0].to(device), inputs[1].to(device), inputs[2]
            targets = targets.to(device)
            
            outputs = model(img_data, tab_data)
            outputs = outputs.squeeze(1)
            
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            targets = targets.cpu().numpy()
            
            # Group predictions and true labels by patient_id
            for patient_id, prob, target in zip(patient_ids, probabilities, targets):
                if patient_id not in y_score_patient:
                    y_score_patient[patient_id] = np.array([prob])  # Initialize as array
                else:
                    y_score_patient[patient_id] = np.append(y_score_patient[patient_id], prob)  # Append to array

                y_true_patient[patient_id] = target 

    # Average scores for each patient
    y_score_patient_avg = {pid: sum(scores) / len(scores) for pid, scores in y_score_patient.items()}
    y_true_patient = {pid: int(label) for pid, label in y_true_patient.items()}

    # Convert to arrays for metric computation
    y_true = list(y_true_patient.values())
    y_score = list(y_score_patient_avg.values())

    # Get best threshold
    best_threshold, precision, recall, thresholds = get_best_threshold(y_true, y_score)
    #best_threshold, _, _, _ = get_best_threshold(y_true, y_score)
    #precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    return best_threshold, precision, recall, thresholds


#TODO: clean utils (eliminate functions not beign used)
def prepare_images(images, labels, general_transform, augmented_tranforms):

    X_augmented = []
    y_augmented = []
    
    for img_path, label in zip(images, labels):
        # Open the image
        #breakpoint()
        image = Image.open(img_path).convert("RGB")
        
        # Conditional augmentation
        if label == 1:  # Example: Flip for label 1
            for i in range(len(augmented_tranforms)):
                augmented_image = augmented_tranforms[i](image)
                X_augmented.append(augmented_image)
                y_augmented.append(label)
        
        image = general_transform(image)
        # Append to the augmented dataset
        X_augmented.append(image)
        y_augmented.append(label)
    
    return X_augmented, y_augmented


class BCEWithLogitsLossLabelSmoothing(nn.Module):
    def __init__(self, alpha=0.1, pos_weight = 2, reduction='mean'):
        """
        BCEWithLogitsLoss with label smoothing.

        Args:
            alpha (float): Smoothing factor. Should be in the range [0,1].
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(BCEWithLogitsLossLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight = pos_weight, reduction='none')  # Compute loss manually

    def forward(self, y_pred, y_true):
        """
        Compute BCEWithLogitsLoss with label smoothing.

        Args:
            y_pred (Tensor): Logits output from the model.
            y_true (Tensor): Ground truth labels (binary, {0,1}).

        Returns:
            Tensor: Loss value.
        """
        # Apply label smoothing
        y_true_smoothed = y_true * (1 - self.alpha) + (1 - y_true) * self.alpha

        # Compute BCEWithLogitsLoss
        loss = self.bce_loss(y_pred, y_true_smoothed)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # 'none'


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # pt is the probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def merge_pre_bn(module, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = module.weight.data
    if module.bias is None:
        zeros = torch.zeros(module.out_channels, device=weight.device).type(weight.type())
        module.bias = nn.Parameter(zeros)
    bias = module.bias.data
    if pre_bn_2 is None:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd
    else:
        assert pre_bn_1.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_1.affine is True, "Unsupport bn_module.affine is False"

        assert pre_bn_2.track_running_stats is True, "Unsupport bn_module.track_running_stats is False"
        assert pre_bn_2.affine is True, "Unsupport bn_module.affine is False"

        scale_invstd_1 = pre_bn_1.running_var.add(pre_bn_1.eps).pow(-0.5)
        scale_invstd_2 = pre_bn_2.running_var.add(pre_bn_2.eps).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight *(pre_bn_1.bias - pre_bn_1.weight * pre_bn_1.running_mean * scale_invstd_1 - pre_bn_2.running_mean) + pre_bn_2.bias

    if isinstance(module, nn.Linear):
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
    elif isinstance(module, nn.Conv2d):
        assert weight.shape[2] == 1 and weight.shape[3] == 1
        weight = weight.reshape(weight.shape[0], weight.shape[1])
        extra_bias = weight @ extra_bias
        weight.mul_(extra_weight.view(1, weight.size(1)).expand_as(weight))
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1, 1)
    bias.add_(extra_bias)

    module.weight.data = weight
    module.bias.data = bias

def cal_flops_params_with_fvcore(model, inputs):
    from fvcore.nn import FlopCountAnalysis, parameter_count_table, parameter_count
    flops = FlopCountAnalysis(model, inputs)
    params = parameter_count(model)
    print('flops(fvcore): %f M' % (flops.total()/1000**2))
    print('params(fvcore): %f M' % (params['']/1000**2))
