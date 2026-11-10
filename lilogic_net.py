# /// script
# dependencies = [
#   "python-dotenv",
#   "numpy",
#   "torch",
#   "torchvision",
#   "tqdm"
# ]
# [tool.uv]
# exclude-newer = "2025-02-20T00:00:00Z"
# ///
# pip install python-dotenv && apt install -y gcc

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split
from datetime import datetime
import time
import os
from zoneinfo import ZoneInfo
import ast

############################ CONFIG ########################

from dotenv import dotenv_values
config = { **dotenv_values(".env"), **os.environ }

DATASET = config.get("DATASET", "MNIST") # MNIST, FashionMNIST, KMNIST, QMNIST, EMNIST, CIFAR10

LOG_TAG = config.get("LOG_TAG", DATASET)
TIMEZONE = config.get("TIMEZONE", "UTC")

IMG_CHANNELS = 3 if DATASET.startswith("CIFAR") else 1
IMG_COUNT = 50_000 if DATASET.startswith("CIFAR") else (240_000 if DATASET == "EMNIST" else 60_000) 
DEFAULT_IMG_WIDTH = 32 if DATASET.startswith("CIFAR") else 28

BINARIZE_IMAGE_THRESHOLD = ast.literal_eval(config.get("BINARIZE_IMAGE_THRESHOLD", "0.75"))
if not isinstance(BINARIZE_IMAGE_THRESHOLD, list):
    BINARIZE_IMAGE_THRESHOLD = [BINARIZE_IMAGE_THRESHOLD] if BINARIZE_IMAGE_THRESHOLD > 0 else []
RGB_TO = config.get("RGB_TO", "MONO") # MONO, RGB, LAB
RGB_TO_RGB = (RGB_TO.startswith("RGB") or RGB_TO == "KEEP")
RGB_TO_YUV = (RGB_TO.startswith("YUV") or RGB_TO.startswith("YCRCB"))
RGB_TO_LAB = (RGB_TO.startswith("LAB") or RGB_TO.startswith("OKLAB"))
BINARIZE_IMAGE_THRESHOLD_DOWNSAMPLE_STEPS = [1,2,2] if RGB_TO.endswith("211") else [1,4,4] if RGB_TO.endswith("411") else [2,1,2] if RGB_TO.endswith("121") else [1] if RGB_TO == "MONO" else [1,1,1]
BINARIZE_IMAGE_WITH_HISTOGRAM = config.get("BINARIZE_IMAGE_WITH_HISTOGRAM", "1" if (IMG_CHANNELS == 1 or RGB_TO_LAB) else "0").lower() in ("true", "1", "yes")
IMG_WIDTH = int(config.get("IMG_WIDTH", DEFAULT_IMG_WIDTH))
IMG_CROP = int(config.get("IMG_CROP", DEFAULT_IMG_WIDTH))
INPUT_SIZE = IMG_WIDTH * IMG_WIDTH * sum([max(1, len(BINARIZE_IMAGE_THRESHOLD[s//2::s])) for s in BINARIZE_IMAGE_THRESHOLD_DOWNSAMPLE_STEPS])
DATA_SPLIT_SEED = int(config.get("DATA_SPLIT_SEED", 42))
TRAIN_FRACTION = float(config.get("TRAIN_FRACTION", 1.0)) # NOTE: since we are not using VALIDATION subset in VALIDATION pass, we can boost accuracy a bit by training on the whole dataset
NUMBER_OF_CATEGORIES = int(config.get("NUMBER_OF_CATEGORIES", 10))

SEED = config.get("SEED", random.randint(0, 1024*1024))
LOG_NAME = f"{LOG_TAG}_{SEED}"
GATE_ARCHITECTURE = ast.literal_eval(config.get("GATE_ARCHITECTURE", "[8000,8000]")) 
INTERCONNECT_ARCHITECTURE = ast.literal_eval(config.get("INTERCONNECT_ARCHITECTURE", "[[],['k',5]]")) 
assert len(GATE_ARCHITECTURE) == len(INTERCONNECT_ARCHITECTURE)
BATCH_SIZE = int(config.get("BATCH_SIZE", 256))

EPOCHS = int(config.get("EPOCHS", 30))
EPOCH_STEPS = math.floor((IMG_COUNT * TRAIN_FRACTION) / BATCH_SIZE)
TRAINING_STEPS = EPOCHS*EPOCH_STEPS
PRINTOUT_EVERY = int(config.get("PRINTOUT_EVERY", EPOCH_STEPS))
VALIDATE_EVERY = int(config.get("VALIDATE_EVERY", EPOCH_STEPS * 25))

LEARNING_RATE = float(config.get("LEARNING_RATE", 0.075))

FORCE_CPU = config.get("FORCE_CPU", "0").lower() in ("true", "1", "yes")
COMPILE_MODEL = config.get("COMPILE_MODEL", "1").lower() in ("true", "1", "yes")

C_INIT = config.get("C_INIT", "NORMAL") # NORMAL, DIRAC
C_SPARSITY = float(config.get("C_SPARSITY", 1.0)) # NOTE: 1.0 works well only for SHALLOW nets, 3.0 for deeper is necessary to binarize well
G_SPARSITY = float(config.get("G_SPARSITY", 1.0))

MANUAL_GAIN = float(config.get("MANUAL_GAIN", 10))

NO_SOFTMAX = config.get("NO_SOFTMAX", "1").lower() in ("true", "1", "yes")

AUGMENTATION_FACTOR = int(config.get("AUGMENTATION_FACTOR", 1))
AUGMENT_NATURAL_IMAGES = config.get("AUGMENT_NATURAL_IMAGES", "0").lower() in ("true", "1", "yes")
AUGMENT_ROTATE = float(config.get("AUGMENT_ROTATE", -1))
AUGMENT_SHEAR = float(config.get("AUGMENT_SHEAR", -1))
AUGMENT_SCALE = float(config.get("AUGMENT_SCALE", -1))
AUGMENT_ELASTIC_ALPHA = float(config.get("AUGMENT_ELASTIC_ALPHA", -1))
AUGMENT_ELASTIC_SIGMA = float(config.get("AUGMENT_ELASTIC_SIGMA", -1))


LEGACY = config.get("LEGACY", "0").lower() in ("true", "1", "yes")

config_printout_keys = ["LOG_NAME", "TIMEZONE", 
               "DATASET", "IMG_COUNT", "RGB_TO",
               "BINARIZE_IMAGE_THRESHOLD", "BINARIZE_IMAGE_WITH_HISTOGRAM",
               "IMG_WIDTH", "INPUT_SIZE", "IMG_CROP", "DATA_SPLIT_SEED", "TRAIN_FRACTION", "NUMBER_OF_CATEGORIES",
               "SEED", "GATE_ARCHITECTURE", "INTERCONNECT_ARCHITECTURE", "BATCH_SIZE",
               "EPOCHS", "EPOCH_STEPS", "TRAINING_STEPS", "PRINTOUT_EVERY", "VALIDATE_EVERY",
               "LEARNING_RATE",
               "C_INIT", "C_SPARSITY", "G_SPARSITY",
               "NO_SOFTMAX",
               "MANUAL_GAIN",
               "FORCE_CPU", "COMPILE_MODEL",
               "AUGMENT_ROTATE", "AUGMENT_SHEAR", "AUGMENT_SCALE", "AUGMENT_ELASTIC_ALPHA", "AUGMENT_ELASTIC_SIGMA",
               "AUGMENTATION_FACTOR", "AUGMENT_NATURAL_IMAGES"
]
config_printout_dict = {key: globals()[key] for key in config_printout_keys}


############################ LOG ########################

def create_logger(timezone):
    def logger(message):
        timestamp = datetime.now(ZoneInfo(timezone))
        print(f'{timestamp.strftime("%H:%M:%S")} {message}', flush=True)
    return logger

log = create_logger(TIMEZONE)

############################ DEVICE ########################

try:
    device = torch.device(
                    "cuda" if torch.cuda.is_available()         and not FORCE_CPU else 
                    "mps"  if torch.backends.mps.is_available() and not FORCE_CPU else 
                    "cpu")
except:
    device = torch.device("cpu")

#################### TENSOR BINARIZATION ##################

def binarize_inplace(x, dim=-1, bin_value=1):
    ones_at = torch.argmax(x, dim=dim)
    x.data.zero_()
    x.data.scatter_(dim=dim, index=ones_at.unsqueeze(dim), value=bin_value)

############################ MODEL ########################

class FixedPowerLawInterconnect(nn.Module):
    def __init__(self, inputs, outputs, alpha, x_min=1.0, name=''):
        super(FixedPowerLawInterconnect, self).__init__()
        self.inputs = inputs
        self.outputs = outputs
        self.alpha = alpha

        max_length = inputs
        size = outputs
        r = torch.rand(size)
        if alpha > 1:
            magnitudes = x_min * (1 - r) ** (-1 / (alpha - 1))          # Power law distribution
            signs = torch.randint(low=0, high=2, size=(size,)) * 2 - 1  # -1 or +1
            offsets = magnitudes * signs * max_length
            indices = torch.arange(start=0, end=size) + offsets.long()
            indices = indices % max_length
        else:
            if LEGACY:
                offsets = r * max_length
                indices = torch.arange(start=0, end=size) + offsets.long()
                indices = indices % max_length
            else:
                c       = torch.randperm(outputs) % inputs
                indices = torch.randperm(inputs)[c]
        self.register_buffer("indices", indices)

    def forward(self, x):
        return x[:, self.indices]

    def __repr__(self):
        with torch.no_grad():
            i = self.indices.view(self.outputs // 2, 2) # [batch_size, number_of_gates, 2]
            A = i[:,0]
            B = i[:,1]

            d = torch.abs(A-B)
            d = torch.minimum(d, self.inputs - d)

            fanout = torch.bincount(self.indices).max().item()
            return f"FixedPowerLawInterconnect({self.inputs} -> {self.outputs // 2}x2, α={self.alpha}, mean={d.float().mean().long()} median={d.float().median().long()} fanout={fanout})"

class DenseInterconnect(nn.Module):
    def __init__(self, inputs, outputs, name=''):
        super(DenseInterconnect, self).__init__()
        self.c = nn.Parameter(torch.zeros((inputs, outputs), dtype=torch.float32))

        if C_INIT == "DIRAC":
            with torch.no_grad():
                nn.init.normal_(self.c, mean=0.0, std=.01) # add a very small amount of noise as a tie-breaker
                idx = torch.randperm(outputs) % inputs
                idx = torch.randperm(inputs)[idx]
                idx = idx.unsqueeze(0)
                self.c.data.scatter_add_(dim=0, index=idx, src=torch.ones_like(idx, dtype=torch.float32) * 10.0) # 10 here is inspired by the min/max range of a gaussian
                                                                                                                 # which for N(0,1) is roughly -5..+5
                                                                                                                 # and it seems to work relatively well with softmax
        else:
            nn.init.normal_(self.c, mean=0.0, std=1)
        self.name = name
        self.binarized = False
    
    def forward(self, x):
        if not self.training or self.binarized:
            indices = torch.argmax(self.c, dim=0)
            return x[:, indices]
        else:
            connections = F.softmax(self.c * C_SPARSITY, dim=0)
            return torch.matmul(x, connections)

    def binarize(self, bin_value=1):
        # TODO: produce final "indices" as well
        binarize_inplace(self.c, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"DenseInterconnect({self.c.shape[0]} -> {self.c.shape[1] // 2}x2)"


class TopKSparseInterconnect(nn.Module):
    def __init__(self, inputs, outputs, topk, name=''):
        super(TopKSparseInterconnect, self).__init__()
        self.top_c = nn.Parameter(torch.zeros((topk, outputs), dtype=torch.float32, device=device))

        with torch.no_grad():
            # NOTE: Dirac does not improve TopK performance, fallback to Gaussian initialisation instead
            cc = torch.normal(mean=0.0, std=1, size=(inputs, outputs), device=device)

            self.top_c.data, top_indices = torch.topk(cc, topk, dim=0, largest=True, sorted=True)
            self.register_buffer("top_indices", top_indices)

        self.inputs = inputs
        self.name = name
        self.binarized = False
    
    def forward(self, x):
        top_indices = self.top_indices
        top_c       = self.top_c

        if not self.training or self.binarized:
            # select from top_indices using top_c
            top1 = torch.argmax(self.top_c, dim=0)
            indices = top_indices[top1, torch.arange(top_indices.shape[-1])]
            return x[:, indices]
        else:
            x = x[:, top_indices]
            top_connections = F.softmax(top_c * C_SPARSITY, dim=0)
            return (x * top_connections).sum(dim=1)

    def binarize(self, bin_value=1):
        top1 = torch.argmax(self.top_c, dim=0)
        indices = self.top_indices[top1, torch.arange(self.top_indices.shape[-1])]
        self.register_buffer("indices", indices)
        
        binarize_inplace(self.top_c, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        percent = self.top_c.shape[0] * 100 / self.inputs
        return f"TopKSparseInterconnect({self.inputs} -> {self.top_c.shape[1] // 2}x2 @ {percent:.2f}%)"

class LearnableGate16Array(nn.Module):
    def __init__(self, number_of_gates, name=''):
        super(LearnableGate16Array, self).__init__()
        self.number_of_gates = number_of_gates
        self.number_of_inputs = number_of_gates * 2
        self.name = name
        self.w = nn.Parameter(torch.zeros((16, self.number_of_gates), dtype=torch.float32)) # [16, W]
        self.binarized = False
        nn.init.normal_(self.w, mean=0.0, std=1)

        # g0  = 
        # g1  =         AB
        # g2  =   A    -AB
        # g3  =   A
        # g4  =      B -AB
        # g5  =      B
        # g6  =   A  B-2AB
        # g7  =   A  B -AB
        # g8  = 1-A -B  AB
        # g9  = 1-A -B 2AB
        # g10 = 1   -B
        # g11 = 1   -B  AB
        # g12 = 1-A
        # g13 = 1-A     AB
        # g14 = 1      -AB
        # g15 = 1

        W = torch.zeros(4, 16, device=device)
        # 1 weights
        W[0, 8:16] =            1
        # A weights
        W[1, [2, 3,  6,  7]] =  1
        W[1, [8, 9, 12, 13]] = -1
        # B weights
        W[2, [4, 5,  6,  7]] =  1
        W[2, [8, 9, 10, 11]] = -1
        # A*B weights
        W[3, 1] =               1
        W[3, 6] =              -2
        W[3, [2, 4, 7,14]] =   -1
        W[3, [1, 8,11,13]] =    1
        W[3, 9] =               2 

        self.W16_to_4 = W


    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.number_of_gates, 2) # [batch_size, number_of_gates, 2]

        A = x[:,:,0]          # [batch_size, number_of_gates]
        B = x[:,:,1]          # [batch_size, number_of_gates]

        if not self.training or self.binarized:
            indices = torch.argmax(self.w, dim=0)
            weights = self.W16_to_4[:, indices]
        else:
            weights = F.softmax(self.w * G_SPARSITY, dim=0)
            weights = torch.matmul(self.W16_to_4, weights) # [4, number_of_gates]
        
        result = weights * torch.stack([torch.ones_like(A), A, B, A*B], dim=1)
        return result.sum(dim=1)

    def binarize(self, bin_value=1):
        # TODO: produce final "indices" as well
        binarize_inplace(self.w, dim=0, bin_value=bin_value)
        self.binarized = True

    def __repr__(self):
        return f"LearnableGate16Array({self.number_of_gates})"

class Model(nn.Module):
    def __init__(self, seed, gate_architecture, interconnect_architecture, number_of_categories, input_size):
        super(Model, self).__init__()
        self.gate_architecture = gate_architecture
        self.interconnect_architecture = interconnect_architecture
        self.first_layer_gates = self.gate_architecture[0]
        self.last_layer_gates = self.gate_architecture[-1]
        self.number_of_categories = number_of_categories
        self.input_size = input_size
        self.seed = seed
        
        self.outputs_per_category = self.last_layer_gates // self.number_of_categories
        assert self.last_layer_gates == self.number_of_categories * self.outputs_per_category

        layers_ = []
        layer_inputs = input_size
        R = [input_size]
        for layer_idx, (layer_gates, interconnect_params) in enumerate(zip(gate_architecture, interconnect_architecture)):
            if   len(interconnect_params) == 2 and (interconnect_params[0] == "k" or interconnect_params[0] == "top_k"):
                interconnect = TopKSparseInterconnect       (layer_inputs, layer_gates*2, topk=        interconnect_params[1],  name=f"i_{layer_idx}")
            elif len(interconnect_params) == 1 and interconnect_params[0] < 0:
                interconnect = FixedPowerLawInterconnect    (layer_inputs, layer_gates*2, alpha=      -interconnect_params[0],  name=f"i_{layer_idx}")
            else:
                interconnect = DenseInterconnect           (layer_inputs, layer_gates*2,                                       name=f"i_{layer_idx}")
            layers_.append(interconnect)
            layers_.append(LearnableGate16Array(layer_gates, f"g_{layer_idx}"))
            layer_inputs = layer_gates
            R.append(layer_gates)
        self.layers = nn.ModuleList(layers_)


    def forward(self, X):
        with torch.no_grad():
            S = torch.sum(X, dim=-1).float()
            self.log_input_mean = torch.mean(S).item()
            self.log_input_std = torch.std(S).item()
            self.log_input_norm = torch.norm(S).item()
        I = X
        R = [I]
        for layer_idx in range(0, len(self.layers)):
            X = self.layers[layer_idx](X)
            if type(self.layers[layer_idx]) is LearnableGate16Array:
                R.append(X)


        X = X.view(X.size(0), self.number_of_categories, self.outputs_per_category).sum(dim=-1)
        if not self.training:   # INFERENCE ends here! Everything past this line will only concern training
            return X            # Finishing inference here is both:
                                # 1) an OPTIMISATION and
                                # 2) it ensures no discrepancy between VALIDATION step during training vs STANDALONE inference

        gain = MANUAL_GAIN
        with torch.no_grad():
            self.log_applied_gain = gain
            self.log_pregain_mean = torch.mean(X).item()
            self.log_pregain_std = torch.std(X).item()
            self.log_pregain_min = torch.min(X).item()
            self.log_pregain_max = torch.max(X).item()
        X = X / gain

        with torch.no_grad():
            self.log_logits_std = torch.std(X).item()
            self.log_logits_norm = torch.norm(X).item()

        if not NO_SOFTMAX:
            X = F.softmax(X, dim=-1)
        return X

    def binarize_inplace_and_prepare_for_export(self, bin_value=1):
        def binarize_if_applicable(layer):
            if hasattr(layer, 'binarize') and callable(layer.binarize):
                layer.binarize(bin_value)
        self.apply(binarize_if_applicable)
        self.eval()

    def get_passthrough_fraction(self):
        pass_fraction_array = []
        indices = torch.tensor([3, 5, 10, 12], dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                pass_fraction_array.append(pass_weight / total_weight)
        return pass_fraction_array
    
    def get_unique_fraction(self):
        unique_fraction_array = []
        with torch.no_grad():
            for model_layer in self.layers:
                # TODO: better to measure only unique indices that point to the previous layer and ignore skip connections:
                # ... = sum(unique_indices < previous_layer.c.shape[1])
                if hasattr(model_layer, 'c'):
                    unique_indices = torch.unique(torch.argmax(model_layer.c, dim=0)).numel()
                    unique_fraction_array.append(unique_indices / model_layer.c.shape[0])
                elif hasattr(model_layer, 'top_c') and hasattr(model_layer, 'top_indices'):
                    outputs = model_layer.top_c.shape[1]
                    top1 = torch.argmax(model_layer.top_c, dim=0)
                    indices = model_layer.top_indices[top1, torch.arange(outputs)]
                    max_inputs = indices.max().item() # approximation
                    unique_indices = torch.unique(indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
                elif hasattr(model_layer, 'indices'):
                    max_inputs = model_layer.indices.max().item() # approximation
                    unique_indices = torch.unique(model_layer.indices).numel()
                    unique_fraction_array.append(unique_indices / max_inputs)
        return unique_fraction_array

    def compute_selected_gates_fraction(self, selected_gates):
        gate_fraction_array = []
        indices = torch.tensor(selected_gates, dtype=torch.long)
        for model_layer in self.layers:
            if hasattr(model_layer, 'w'):
                weights_after_softmax = F.softmax(model_layer.w, dim=0)
                pass_weight = (weights_after_softmax[indices, :]).sum()
                total_weight = weights_after_softmax.sum()
                gate_fraction_array.append(pass_weight / total_weight)
        return torch.mean(torch.tensor(gate_fraction_array)).item()


### INSTANTIATE THE MODEL AND MOVE TO GPU ###

log(f"PREPARE MODEL on device={device}")
random.seed(SEED)
torch.manual_seed(SEED)
model = Model(SEED, GATE_ARCHITECTURE, INTERCONNECT_ARCHITECTURE, NUMBER_OF_CATEGORIES, INPUT_SIZE).to(device)
if COMPILE_MODEL:
    torch.set_float32_matmul_precision('high')
    model = torch.compile(model)
log(f"model={model}")

############################ DATA ########################

### GENERATORS
def transform(apply_augmentations=False):
    def srgb_to_linear(x):
        return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    def linear_to_srgb(x):
        return torch.where(x <= 0.0031308, 12.92 * x, 1.055 * torch.pow(x, 1/2.4) - 0.055)

    def srgb_to_oklab(x):
        # x: 3xHxW in [0,1] sRGB
        if x.dim() == 3: x = x.unsqueeze(0) # 1x3xHxW
        r, g, b = x[:,0:1], x[:,1:2], x[:,2:3]
        r, g, b = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)

        # M1: linear RGB -> LMS (OKLab)
        l = 0.4122214708*r + 0.5363325363*g + 0.0514459929*b
        m = 0.2119034982*r + 0.6806995451*g + 0.1073969566*b
        s = 0.0883024619*r + 0.2817188376*g + 0.6299787005*b

        l_, m_, s_ = l.clamp_min(0).pow(1/3), m.clamp_min(0).pow(1/3), s.clamp_min(0).pow(1/3)

        # M2: LMS^(1/3) -> OKLab
        L =  0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
        a =  1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
        b =  0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_

        out = torch.cat([linear_to_srgb(L), a, b], dim=1) # 1x3xHxW
        return out.squeeze(0) # 3xHxW

    def srgb_to_ycrcb(x):
        if x.dim() == 3: x = x.unsqueeze(0)  # 1x3xHxW
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]

        # Standard BT.601 (full-range) coefficients
        # Reference: https://www.itu.int/rec/R-REC-BT.601
        Y  = 0.299 * r + 0.587 * g + 0.114 * b
        Cb = (b - Y) * 0.564 + 0.5
        Cr = (r - Y) * 0.713 + 0.5

        out = torch.cat([Y, Cr, Cb], dim=1) # 1x3xHxW
        return out.squeeze(0) # 3xHxW

    def binarize_image(image, thresholds=BINARIZE_IMAGE_THRESHOLD, downsample_thresholds=BINARIZE_IMAGE_THRESHOLD_DOWNSAMPLE_STEPS, use_histogram=BINARIZE_IMAGE_WITH_HISTOGRAM):
        if image.shape[0] == 3: # 3 chanel image
            assert image.dim() == 3
            image = image.unsqueeze(0)
            r, g, b = image[:,0:1], image[:,1:2], image[:,2:3]
            ds = downsample_thresholds
            return torch.cat([
                binarize_image(r, thresholds[ds[0]//2::ds[0]], downsample_thresholds=[1], use_histogram=use_histogram),
                binarize_image(g, thresholds[ds[1]//2::ds[1]], downsample_thresholds=[1], use_histogram=use_histogram),
                binarize_image(b, thresholds[ds[2]//2::ds[2]], downsample_thresholds=[1], use_histogram=use_histogram)]).view(-1)

        assert image.dim() == 1 or image.shape[0] == 1
        assert isinstance(thresholds, list)
        thresholds = torch.as_tensor(thresholds, dtype=image.dtype, device=image.device)
        if use_histogram:
            thresholds = torch.quantile(image, thresholds) # use quantiles as thresholds
        # channels = [(image > q).float() for q in thresholds]
        channels = [(image > q).bool() for q in thresholds]
        return torch.stack(channels, dim=0).view(-1)

    def apply_transform(apply_augmentations=False):
        augmentations = []
        if apply_augmentations:
            if AUGMENT_ROTATE>0 or AUGMENT_SHEAR>0 or AUGMENT_SCALE>0:
                augmentations.append(
                    transforms.RandomAffine(degrees=AUGMENT_ROTATE,
                                            scale=(1-AUGMENT_SCALE,1+AUGMENT_SCALE),
                                            shear=AUGMENT_SHEAR))
            if AUGMENT_ELASTIC_ALPHA>0 or AUGMENT_ELASTIC_SIGMA>0:
                augmentations.append(
                    transforms.ElasticTransform(alpha=AUGMENT_ELASTIC_ALPHA, sigma=AUGMENT_ELASTIC_SIGMA))
            if AUGMENT_NATURAL_IMAGES:
                augmentations.append(transforms.RandomCrop(IMG_CROP, padding=4, padding_mode='reflect'))
                augmentations.append(transforms.RandomHorizontalFlip())

        convert_colors = []
        if IMG_CHANNELS == 1:
            convert_colors = [transforms.Lambda(lambda x: x.view(-1))]
        elif IMG_CHANNELS == 3:
            if RGB_TO_RGB:
                pass
            elif RGB_TO_LAB:
                convert_colors = [transforms.Lambda(srgb_to_oklab)]
            elif RGB_TO_YUV:
                convert_colors = [transforms.Lambda(srgb_to_ycrcb)]
            else:
                convert_colors = [transforms.Grayscale(1)]
        else:
            raise "Not supported image format!"

        binarize_inputs = []
        if len(BINARIZE_IMAGE_THRESHOLD) > 0:
            binarize_inputs = [transforms.Lambda(lambda x: binarize_image(x))]

        return transforms.Compose([
            *augmentations,
            transforms.CenterCrop((IMG_CROP, IMG_CROP)),
            transforms.Resize((IMG_WIDTH, IMG_WIDTH)),
            transforms.ToTensor(),
            *convert_colors,
            *binarize_inputs,
            transforms.Lambda(lambda x: x.view(-1))
        ])

    return apply_transform(apply_augmentations)

import inspect
import hashlib
import re
    
def get_code_hash(fn, verbose=False):
    #n=0,extra_variables=['AUGMENT_ROTATE','AUGMENT_SHEAR','AUGMENT_SCALE','AUGMENT_ELASTIC_SIGMA','AUGMENT_ELASTIC_ALPHA','BINARIZE_IMAGE_TRESHOLD','DATA_SPLIT_SEED','INPUT_SIZE']):
    def replace_globals_in_string(s: str) -> str:
        for name, value in globals().items():
            if inspect.isfunction(value) or inspect.isclass(value) or inspect.ismodule(value):
                continue
            try:
                pattern = re.escape(name)
                replacement = str(value)
                s = re.sub(rf'\b{pattern}\b', replacement, s)
            except Exception:
                continue
        return s
    src = inspect.getsource(fn)
    src = replace_globals_in_string(src)

    if verbose: log(src)
    return hashlib.sha1(src.encode('utf-8')).hexdigest()[:10]

log(f"READ DATA")

def load_or_build_cached_dataset(name, root, train, transform, hash, augmentation_factor=1):
    split = 'train' if train else 'test'
    path = f'{root}/cached_{name.lower()}_{split}_x{augmentation_factor}_{hash}.pt' #  get_cache_path(config, split)
    try:
        cached = torch.load(path, weights_only=False)
        print(f"Loaded cached {name.upper()} {split} from {path}")
        return torch.utils.data.TensorDataset(cached['data'], cached['labels'])
    except FileNotFoundError:
        print(f"No cache found for {name.upper()} {split} with hash {hash}. Building new cache at {path}...")
        dataset_class = getattr(torchvision.datasets, name)
        extra_args = {"split": "digits"} if name == "EMNIST" else {} # EMNIST is a special case as it has multiple splits
                                                                     # we assume only 'digits' subset with 10 classes
                                                                     # similar to the original MNIST here
        inputs, labels = [], []
        print(f"Loading dataset from {dataset_class}")

        from tqdm import tqdm

        for aug_idx in range(max(1, augmentation_factor)):
            dataset = dataset_class(
                root=root,
                train=train,
                transform=transform,
                download=True,
                **extra_args
            )

            for x, y in tqdm(
                dataset,
                desc=f"{name} {split} augmentation {aug_idx+1}/{augmentation_factor}",
                total=len(dataset)
            ):
                inputs.append(x.unsqueeze(0))
                labels.append(torch.tensor(y).unsqueeze(0))

        data_tensor = torch.cat(inputs, dim=0)
        label_tensor = torch.cat(labels, dim=0)
        torch.save({'data': data_tensor, 'labels': label_tensor, 'hash': hash}, path)
        print(f"{name} {split} dataset contains {len(inputs)} inputs and {len(torch.unique(label_tensor))} classes")
        return torch.utils.data.TensorDataset(data_tensor, label_tensor)

# NOTE: It is not simple to calculate hash for code path that does transformation without augmentation
# that means that validation & test will be rebuild everytime augmentation parameters change
# but it avoids otherwise unnecessary code complication
hash_transform_fn = get_code_hash(transform)#, verbose=True)
train_dataset = load_or_build_cached_dataset(
    name=DATASET, # MNIST, CIFAR10 ...
    root="./data",
    train=True,
    transform=transform(apply_augmentations=(AUGMENTATION_FACTOR > 0)),
    hash=hash_transform_fn,
    augmentation_factor=AUGMENTATION_FACTOR
)

test_dataset = load_or_build_cached_dataset(
    name=DATASET, # MNIST, CIFAR10 ...
    root="./data",
    train=False,
    transform=transform(apply_augmentations=False),
    hash=hash_transform_fn
)

### SPLIT TRAIN DATASET ###

log(f"UPLOAD DATA")
train_size = int(TRAIN_FRACTION * len(train_dataset))
val_size = len(train_dataset) - train_size
if TRAIN_FRACTION < 1.0:
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(DATA_SPLIT_SEED))

### MOVE TRAIN DATASET TO GPU ###

train_images_storage_dtype = torch.bool if len(BINARIZE_IMAGE_THRESHOLD) > 0 else torch.float16
test_images_storage_dtype = val_images_storage_dtype = torch.float16

train_dataset_samples = len(train_dataset)
train_images = torch.empty((train_dataset_samples, INPUT_SIZE), dtype=train_images_storage_dtype)
train_labels = torch.empty((train_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

train_labels_ = torch.empty((train_dataset_samples), dtype=torch.long)
for i, (image, label) in enumerate(train_dataset):
    train_images[i] = image
    train_labels_[i] = label
train_labels = torch.nn.functional.one_hot(train_labels_, num_classes=NUMBER_OF_CATEGORIES)
train_labels = train_labels.type(torch.float32)

train_images = train_images.to(device)
train_labels = train_labels.to(device)

### MOVE VAL DATASET TO GPU ###

if TRAIN_FRACTION < 1.0:
    val_dataset_samples = len(val_dataset)

    val_images = torch.empty((val_dataset_samples, INPUT_SIZE), dtype=val_images_storage_dtype)
    val_labels = torch.empty((val_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

    val_labels_ = torch.empty((val_dataset_samples), dtype=torch.long)
    for i, (image, label) in enumerate(val_dataset):
        val_images[i] = image
        val_labels_[i] = label
    val_labels = torch.nn.functional.one_hot(val_labels_, num_classes=NUMBER_OF_CATEGORIES)
    val_labels = val_labels.type(torch.float32)

    val_images = val_images.to(device)
    val_labels = val_labels.to(device)

### MOVE TEST DATASET TO GPU ###

test_dataset_samples = len(test_dataset)

test_images = torch.empty((test_dataset_samples, INPUT_SIZE), dtype=test_images_storage_dtype)
test_labels = torch.empty((test_dataset_samples, NUMBER_OF_CATEGORIES), dtype=torch.float32)

test_labels_ = torch.empty((test_dataset_samples), dtype=torch.long)
for i, (image, label) in enumerate(test_dataset):
    test_images[i] = image
    test_labels_[i] = label
test_labels = torch.nn.functional.one_hot(test_labels_, num_classes=NUMBER_OF_CATEGORIES)
test_labels = test_labels.type(torch.float32)

test_images = test_images.to(device)
test_labels = test_labels.to(device)

### VALIDATE ###

def get_validate(default_model):
    def validate(dataset="train", model=default_model):
        if dataset == "val":
            number_of_samples = val_dataset_samples
            sample_images = val_images
            sample_labels = val_labels
        elif dataset == "test":
            number_of_samples = test_dataset_samples
            sample_images = test_images
            sample_labels = test_labels
        elif dataset == "train":
            number_of_samples = train_dataset_samples
            sample_images = train_images
            sample_labels = train_labels
        else:
            raise IOError(f"Unknown dataset {dataset}")
        val_loss = 0.0
        val_steps = 0
        correct = 0
        for start_idx in range(0, number_of_samples, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, number_of_samples)
            val_indices = torch.arange(start_idx, end_idx, device=device)    
            x_val = sample_images[val_indices].float()
            y_val = sample_labels[val_indices]
            with torch.no_grad():
                val_output = model(x_val)
                val_loss += F.cross_entropy(val_output, y_val, reduction="sum").item()
                correct += (val_output.argmax(dim=1) == y_val.argmax(dim=1)).sum().item()
            val_steps += len(x_val)
        assert val_steps == number_of_samples
        val_loss /= val_steps
        val_accuracy = correct / val_steps
        return val_loss, val_accuracy
    return validate

### TRAIN ###

validate = get_validate(model)
passthrough_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_passthrough_fraction()])
unique_log = ", ".join([f"{value * 100:4.1f}%" for value in model.get_unique_fraction()])

log(f"EPOCH_STEPS={EPOCH_STEPS}, will train for {EPOCHS} EPOCHS")
if LEGACY:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0) # if weight decay encourages uniform distribution
else:
    optimizer = torch.optim.Adam (model.parameters(), lr=LEARNING_RATE)
time_start = time.time()

for i in range(TRAINING_STEPS):
    if LEGACY:
        indices = torch.randint(0, train_dataset_samples, (BATCH_SIZE,), device=device)
    else:
        # shuffle dataset instead of random sampling
        start_idx = (i * BATCH_SIZE) % train_dataset_samples
        end_idx = min(start_idx + BATCH_SIZE, train_dataset_samples)
        if start_idx < BATCH_SIZE: # reshuffle indices every time we reached the end of the dataset

            # and start_idx has wrapped around
            all_indices = torch.randperm(train_dataset_samples, device=device)
        indices = all_indices[start_idx: end_idx]    

    x = train_images[indices].float()
    y = train_labels[indices]

    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        model_output = model(x)
        loss = F.cross_entropy(model_output, y)
        loss.backward()
        optimizer.step()

    def compute_decay_rate(start_value, end_value, steps):
        return np.exp(np.log(end_value / start_value) / steps)

    if (i + 1) % PRINTOUT_EVERY == 0:
        passthrough_log = " ".join([f"{value * 100:2.0f}" for value in model.get_passthrough_fraction()])
        unique_log = " ".join([f"{value * 100:2.0f}" for value in model.get_unique_fraction()])
        grad_log = " ".join([f"{torch.log10(value.grad.std())*10:2.0f}" for value in model.parameters()])
        logits_log = f"{model.log_pregain_min:.0f}..{model.log_pregain_max:.0f} μ{model.log_pregain_mean:.0f}±{model.log_pregain_std:.1f}/{model.log_applied_gain:.1f} = ±{model.log_logits_std:.1f} ‖{model.log_logits_norm:.0f}‖"
        log(f"Iteration {i + 1:10} - Loss {loss:6.3f} - Pass (%) {passthrough_log} | Conn (%) {unique_log} | e-∇x10 {grad_log} | Logits {logits_log}")
    if (i + 1) % VALIDATE_EVERY == 0 or (i + 1) == TRAINING_STEPS:
        current_epoch = (i+1) // EPOCH_STEPS

        train_loss, train_acc = validate('train')
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TRN loss={train_loss:.3f} acc={train_acc*100:.2f}%")

        if TRAIN_FRACTION < 1.0:
            val_loss, val_acc = validate('val')
            log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     VAL loss={val_loss:.3f} acc={val_acc*100:.2f}%")

        model.eval() # switch to binarized evaluation mode

        _, bin_train_acc = validate(dataset="train")
        bin_train_diff = train_acc - bin_train_acc
        log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS} BIN TRN            acc={bin_train_acc*100:.2f}%, bin_train_diff={bin_train_diff*100:.2f}%")

        if TRAIN_FRACTION < 1.0:
            _, bin_val_acc = validate(dataset="val")
            bin_val_diff = val_acc - bin_val_acc
            log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS} BIN VAL            acc={bin_val_acc*100:.2f}%, bin_train_diff={bin_val_diff*100:.2f}%")

        model.train() # switch back to floating point evaluation mode

        if TRAIN_FRACTION == 1.0:
            test_loss, test_acc = validate('test')
            test_acc_diff = train_acc - test_acc
            log(f"{LOG_NAME} EPOCH={current_epoch}/{EPOCHS}     TST            acc={test_acc*100:.2f}%, test_acc_diff= {test_acc_diff*100:.2f}%, loss={test_loss:.3f}")


time_end = time.time()
training_total_time = time_end - time_start 
log(f"Training took {training_total_time:.2f} seconds, per iteration: {(training_total_time) / TRAINING_STEPS * 1000:.2f} milliseconds")

test_loss, test_acc = validate('test')
log(f"    TEST loss={test_loss:.3f} acc={test_acc*100:.2f}%")


model.binarize_inplace_and_prepare_for_export()
bin_test_loss, bin_test_acc = validate(dataset="test")
log(f"BIN TEST loss={bin_test_loss:.3f} acc={bin_test_acc*100:.2f}%")
