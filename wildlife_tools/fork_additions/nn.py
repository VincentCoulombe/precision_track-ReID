import math
import os
from abc import ABC, abstractmethod
from itertools import chain

import timm
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from transformers import CLIPModel, CLIPProcessor

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _warmup_cosine_lambda(warmup_epochs: int, total_epochs: int, min_factor: float = 1e-3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(min_factor, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return lr_lambda


def _split_params(model: nn.Module, objective):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = (
        list(model.reduction_layer.parameters())
        + list(model.head.parameters())
        + list(objective.arcface_loss.parameters())
    )
    return backbone_params, head_params


class BaseBackbone(ABC):
    @abstractmethod
    def create_backbone(self, pretrained: bool) -> nn.Module:
        """Create and return the backbone model."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension of the backbone."""
        pass

    @abstractmethod
    def get_processor(self):
        """Return the processor/transform for this backbone (if any)."""
        pass

    @abstractmethod
    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the backbone given input tensor."""
        pass

    @abstractmethod
    def get_train_transforms(self, img_size) -> T.Compose:
        """Return the per-publication training augmentation pipeline."""
        pass

    @abstractmethod
    def get_test_transforms(self, img_size) -> T.Compose:
        """Return the per-publication evaluation preprocessing pipeline."""
        pass

    @abstractmethod
    def build_optimizer(self, model: nn.Module, objective) -> torch.optim.Optimizer:
        """Build the optimizer using the per-publication recipe."""
        pass

    @abstractmethod
    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        """Build the LR scheduler using the per-publication recipe."""
        pass


class MegaDescriptor(BaseBackbone):
    def __init__(self, model_name: str = "hf-hub:BVRA/MegaDescriptor-T-224"):
        self.model_name = model_name

    def create_backbone(self, pretrained: bool) -> nn.Module:
        return timm.create_model(self.model_name, num_classes=0, pretrained=pretrained)

    def get_embedding_dim(self) -> int:
        return 768

    def get_processor(self):
        return None

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return backbone(x)

    def get_train_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
                T.RandAugment(num_ops=2, magnitude=20),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_test_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def build_optimizer(self, model: nn.Module, objective) -> torch.optim.Optimizer:
        params = chain(
            filter(lambda p: p.requires_grad, model.parameters()),
            objective.arcface_loss.parameters(),
        )
        return SGD(params=params, lr=1e-3, momentum=0.9)

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        warmup_epochs = int(2 * epochs / 3)
        cosine_epochs = epochs - warmup_epochs
        min_lr = optimizer.defaults.get("lr") * 1e-3
        warmup = LambdaLR(optimizer, lr_lambda=lambda epoch: (epoch + 1) / warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=min_lr)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


class CLIP(BaseBackbone):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._processor = None

    def create_backbone(self, pretrained: bool) -> nn.Module:
        if pretrained:
            clip_model = CLIPModel.from_pretrained(self.model_name)
        else:
            from transformers import CLIPConfig

            config = CLIPConfig.from_pretrained(self.model_name)
            clip_model = CLIPModel(config)
        self._vision_model = clip_model.vision_model
        return self._vision_model

    def get_embedding_dim(self) -> int:
        return self._vision_model.config.hidden_size

    def get_processor(self):
        if self._processor is None:
            self._processor = CLIPProcessor.from_pretrained(self.model_name)
        return self._processor

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        outputs = backbone(x)
        return outputs.pooler_output

    def get_train_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(size=img_size, scale=(0.9, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def get_test_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def build_optimizer(self, model: nn.Module, objective) -> torch.optim.Optimizer:
        backbone_params, head_params = _split_params(model, objective)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": 1e-5})
        groups.append({"params": head_params, "lr": 1e-3})
        return AdamW(groups, weight_decay=0.1, betas=(0.9, 0.98))

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        return LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lambda(warmup_epochs=10, total_epochs=epochs),
        )


class DINOv3(BaseBackbone):
    model_name: str
    backbone_lr: float
    head_lr: float
    weight_decay: float

    def create_backbone(self, pretrained: bool) -> nn.Module:
        self._backbone = timm.create_model(self.model_name, num_classes=0, pretrained=pretrained)
        return self._backbone

    def get_embedding_dim(self) -> int:
        return self._backbone.num_features

    def get_processor(self):
        return None

    def extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return backbone(x)

    def get_train_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomResizedCrop(size=img_size, scale=(0.8, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandAugment(num_ops=2, magnitude=5),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.03),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_test_transforms(self, img_size) -> T.Compose:
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def build_optimizer(self, model: nn.Module, objective) -> torch.optim.Optimizer:
        backbone_params, head_params = _split_params(model, objective)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.backbone_lr})
        groups.append({"params": head_params, "lr": self.head_lr})
        return AdamW(groups, weight_decay=self.weight_decay, betas=(0.9, 0.999))

    def build_scheduler(self, optimizer: torch.optim.Optimizer, epochs: int):
        return LambdaLR(
            optimizer,
            lr_lambda=_warmup_cosine_lambda(warmup_epochs=10, total_epochs=epochs),
        )


class DINOv3ViT(DINOv3):
    model_name = "timm/vit_small_patch16_dinov3.lvd1689m"
    backbone_lr = 1e-5
    head_lr = 1e-3
    weight_decay = 0.04


class DINOv3ConvNeXt(DINOv3):
    model_name = "timm/convnext_tiny.dinov3_lvd1689m"
    backbone_lr = 4e-5
    head_lr = 1e-3
    weight_decay = 1e-4


BACKBONE = {
    "megadescriptor": MegaDescriptor,
    "clip": CLIP,
    "dinov3-vit": DINOv3ViT,
    "dinov3-convnext": DINOv3ConvNeXt,
}


class PtReIDModel(nn.Module):
    def __init__(self, config, checkpoint=None, pretrained=False):
        super().__init__()
        backbone_name = config.backbone_name
        if backbone_name.lower() not in BACKBONE:
            raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(BACKBONE.keys())}")

        self.strategy = BACKBONE[backbone_name.lower()]()
        self.backbone = self.strategy.create_backbone(pretrained)

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.reduction_layer = nn.Linear(self.strategy.get_embedding_dim(), config.n_output_embd)
        self.head = ClassifierHead(config)

        if checkpoint is not None:
            assert os.path.isfile(checkpoint), f"The provided checkpoint: '{checkpoint}' does not exists."

            state_dict = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            self.load_state_dict(state_dict["model"])

        self._return_features = True
        self._return_logits = True

    @property
    def return_features(self):
        return self._return_features

    @return_features.setter
    def return_features(self, activate):
        assert isinstance(activate, bool)
        self._return_features = activate

    @property
    def return_logits(self):
        return self._return_logits

    @return_logits.setter
    def return_logits(self, activate):
        assert isinstance(activate, bool)
        self._return_logits = activate

    @property
    def nb_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_processor(self):
        """Return the processor for the current backbone strategy."""
        return self.strategy.get_processor()

    def forward(self, x):
        features = self.strategy.extract_features(self.backbone, x)
        reduced_features = self.reduction_layer(features)
        logits = self.head(reduced_features)

        if self._return_logits and self._return_features:
            return reduced_features, logits
        elif self._return_features:
            return reduced_features
        elif self._return_logits:
            return logits


class ClassifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.blocks = [MLP(config) for _ in range(config.n_layers)]
        self.blocks.append(nn.Linear(config.n_output_embd, config.n_classes))
        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_output_embd)
        self.c_fc = nn.Linear(config.n_output_embd, 4 * config.n_output_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_output_embd, config.n_output_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        h = self.ln(x)
        h = self.c_fc(h)
        h = self.gelu(h)
        h = self.c_proj(h)
        h = self.dropout(h)
        return x + h
