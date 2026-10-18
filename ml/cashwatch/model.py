"""A trainable image encoder plus temporal convolutional classifier."""
from __future__ import annotations

import copy
import torch
from torch import nn
import timm

MODEL_NAME = "mobilenetv3_small_100.lamb_in1k"


class VideoDetector(nn.Module):
    def __init__(self, pretrained=False, head_type="temporal", action_classes=0, point_localizer=False):
        super().__init__()
        self.encoder = timm.create_model(MODEL_NAME, pretrained=pretrained, num_classes=0)
        features = self.encoder.conv_head.out_channels
        self.head_type = head_type
        self.normalization = nn.LayerNorm(features)
        self.temporal = nn.Sequential(
            nn.Conv1d(features * 2, 96, kernel_size=3, padding=1), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(96, 32, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(32, 1))
        if head_type == "linear":
            self.temporal = nn.Linear(features, 1)
        elif head_type != "temporal":
            raise ValueError("Unknown detector head")
        self.action_classes = int(action_classes)
        if self.action_classes:
            self.action_head = nn.Sequential(nn.Linear(features * 2, 128), nn.GELU(),
                nn.Dropout(0.35), nn.Linear(128, self.action_classes))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if point_localizer:
            # Separate point-supervised head; existing detector weights and API stay intact.
            self.localizer = copy.deepcopy(self.temporal)

    def configure_finetuning(self, unfreeze_encoder=False):
        for param in self.encoder.parameters():
            param.requires_grad_(False)
        if unfreeze_encoder:
            for module in [self.encoder.blocks[-2:], self.encoder.conv_head]:
                for param in module.parameters():
                    param.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.encoder.parameters()):
            self.encoder.eval()
        # Tiny, correlated clip batches should not destroy pretrained BN statistics.
        for module in self.encoder.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def encode(self, frames):
        return self.encoder((frames - self.mean) / self.std)

    def classify(self, embeddings):
        features = self.normalization(embeddings)
        if self.head_type == "linear":
            return self.temporal(features.mean(dim=1)).squeeze(-1)
        difference = torch.zeros_like(features)
        difference[:, 1:] = features[:, 1:] - features[:, :-1]
        return self.temporal(torch.cat([features, difference], dim=-1).transpose(1, 2)).squeeze(-1)

    def forward(self, clips):
        batch, length, channels, height, width = clips.shape
        features = self.encode(clips.reshape(batch * length, channels, height, width))
        return self.classify(features.reshape(batch, length, -1))

    def localize(self, embeddings):
        """Predict proximity to an annotated event center from visual embeddings."""
        if not hasattr(self, "localizer"):
            raise ValueError("This checkpoint has no trained point localizer")
        features = self.normalization(embeddings)
        if self.head_type == "linear":
            return self.localizer(features.mean(dim=1)).squeeze(-1)
        difference = torch.zeros_like(features)
        difference[:, 1:] = features[:, 1:] - features[:, :-1]
        return self.localizer(torch.cat([features, difference], dim=-1).transpose(1, 2)).squeeze(-1)

    def classify_actions(self, embeddings):
        """Supervised visual action logits; no filename or annotation inputs."""
        if not self.action_classes:
            raise ValueError("This checkpoint has no trained action classifier")
        features = self.normalization(embeddings)
        motion = (features[:, 1:] - features[:, :-1]).abs().mean(dim=1)
        return self.action_head(torch.cat([features.mean(dim=1), motion], dim=-1))


def select_device(requested="auto"):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(requested)
