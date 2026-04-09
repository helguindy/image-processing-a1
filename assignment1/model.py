import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

import config


def get_model(in_channels=3):
    """
    Returns a MobileNetV2 model ready for CIFAR-10 classification.

    in_channels : number of channels coming out of the domain transform
                  - Spatial  → 3  (normal RGB)
                  - Fourier  → 3  (log-magnitude per channel)
                  - Wavelet  → 12 (4 subbands × 3 channels) or 3 (LL only)
                  - Custom   → depends on the chosen domain
    """

    # Load MobileNetV2 with ImageNet pretrained weights
    # Pretrained weights help — the model already knows edges, textures, shapes
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    # ── Adapt the first layer if input channels differ from 3 ─────────────────
    # MobileNetV2 normally expects 3-channel (RGB) input.
    # For domains with a different number of channels, we replace the first conv.
    if in_channels != 3:
        first_conv = model.features[0][0]   # original Conv2d(3, 32, ...)
        model.features[0][0] = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )

    # ── Replace the final classifier for CIFAR-10 (10 classes) ───────────────
    # Original MobileNetV2 outputs 1000 classes (ImageNet).
    # We replace it with a layer that outputs config.NUM_CLASSES (10).
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config.NUM_CLASSES)

    return model
