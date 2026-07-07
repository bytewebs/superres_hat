from typing import Any, cast, Dict, List, Union

import os
import torch
import functools
from torch import nn, Tensor
from torch.nn import functional as F_torch
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names


feature_extractor_net_cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(net_cfg_name: str, batch_norm: bool = False) -> nn.Sequential:
    net_cfg = feature_extractor_net_cfgs[net_cfg_name]
    layers: nn.Sequential[nn.Module] = nn.Sequential()
    in_channels = 3
    for v in net_cfg:
        if v == "M":
            layers.append(nn.MaxPool2d((2, 2), (2, 2)))
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, (3, 3), (1, 1), (1, 1))
            if batch_norm:
                layers.append(conv2d)
                layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(True))
            else:
                layers.append(conv2d)
                layers.append(nn.ReLU(True))
            in_channels = v

    return layers


class _FeatureExtractor(nn.Module):
    def __init__(
            self,
            net_cfg_name: str = "vgg19",
            batch_norm: bool = False,
            num_classes: int = 1000) -> None:
        super(_FeatureExtractor, self).__init__()
        self.features = _make_layers(net_cfg_name, batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

        # Initialize neural network weights
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class ContentLoss(torch.nn.Module):
    """VGG19 content/perceptual loss.

    Supports the two calling conventions that coexist in this repository:

    * Training (and ESRGAN-PA eval) -- 8 positional args::

          ContentLoss(net_cfg_name, batch_norm, num_classes, model_weights_path,
                      feature_nodes, feature_weights,
                      feature_normalize_mean, feature_normalize_std, use_huber=...)

      Taps intermediate activations by node name via ``create_feature_extractor``
      (this is the implementation prior training runs were written against).

    * Evaluation -- 4 positional args::

          ContentLoss(model_weights_path, feature_weights,
                      feature_normalize_mean, feature_normalize_std, use_huber=...)

      Taps fixed ``vgg19.features`` slices.
    """

    def __init__(self, *args, use_huber=True):
        super(ContentLoss, self).__init__()

        if len(args) >= 8:
            (net_cfg_name, batch_norm, num_classes, model_weights_path,
             feature_nodes, feature_weights,
             feature_normalize_mean, feature_normalize_std) = args[:8]
            self._mode = "nodes"

            # Define the feature extraction model
            model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
            # Load the pre-trained model
            if model_weights_path is None:
                model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            elif model_weights_path is not None and os.path.exists(model_weights_path):
                checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
                if "state_dict" in checkpoint.keys():
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError("Model weight file not found")

            self.feature_extractor = create_feature_extractor(model, feature_nodes)
            self.feature_extractor_nodes = feature_nodes
            for model_parameters in self.feature_extractor.parameters():
                model_parameters.requires_grad = False
            self.feature_extractor.eval()

        elif len(args) == 4:
            model_weights_path, feature_weights, feature_normalize_mean, feature_normalize_std = args
            self._mode = "blocks"

            if model_weights_path is None:
                model = models.vgg19(pretrained=True)
            elif model_weights_path is not None and os.path.exists(model_weights_path):
                checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
                if "state_dict" in checkpoint.keys():
                    model.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint)
            else:
                raise FileNotFoundError("Model weight file not found")

            blocks = []
            blocks.append(model.features[:2].eval())
            blocks.append(model.features[2:7].eval())
            blocks.append(model.features[7:16].eval())
            blocks.append(model.features[16:25].eval())
            blocks.append(model.features[25:34].eval())
            for bl in blocks:
                for p in bl.parameters():
                    p.requires_grad = False
            self.blocks = torch.nn.ModuleList(blocks)
        else:
            raise TypeError(
                f"ContentLoss expected 4 (eval) or 8 (training) positional args, got {len(args)}.")

        self.feature_weights = feature_weights
        self.use_huber = use_huber
        self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)

    def forward(self, input, target):
        assert input.size() == target.size(), "Two tensor must have the same size"
        device = input.device

        # input normalization
        input = self.normalize(input)
        target = self.normalize(target)

        losses = []
        if self._mode == "blocks":
            x = input
            y = target
            for i, block in enumerate(self.blocks):
                x = block(x)
                y = block(y)
                if not self.use_huber:
                    losses.append(self.feature_weights[i]*F_torch.l1_loss(x, y))
                else:
                    losses.append(self.feature_weights[i]*F_torch.huber_loss(x, y, delta=2))
        else:
            input_features = self.feature_extractor(input)
            target_features = self.feature_extractor(target)
            for i, node in enumerate(self.feature_extractor_nodes):
                if not self.use_huber:
                    losses.append(self.feature_weights[i]*F_torch.l1_loss(input_features[node], target_features[node]))
                else:
                    losses.append(self.feature_weights[i]*F_torch.huber_loss(input_features[node], target_features[node], delta=2))

        losses = torch.Tensor([losses]).to(device)
        return losses


# class ContentLoss(nn.Module):
#     """Constructs a content loss function based on the VGG19 network.
#     Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

#     Paper reference list:
#         -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
#         -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
#         -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

#      """

#     def __init__(
#             self,
#             net_cfg_name: str,
#             batch_norm: bool,
#             num_classes: int,
#             model_weights_path: str,
#             feature_nodes: list,
#             feature_weights: list,
#             feature_normalize_mean: list,
#             feature_normalize_std: list,
#             use_huber = True,
#     ) -> None:
#         super(ContentLoss, self).__init__()
#         # Define the feature extraction model
#         model = _FeatureExtractor(net_cfg_name, batch_norm, num_classes)
#         # Load the pre-trained model
#         if model_weights_path is None:
#             model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
#         elif model_weights_path is not None and os.path.exists(model_weights_path):
#             checkpoint = torch.load(model_weights_path, map_location=lambda storage, loc: storage)
#             if "state_dict" in checkpoint.keys():
#                 model.load_state_dict(checkpoint["state_dict"])
#             else:
#                 model.load_state_dict(checkpoint)
#         else:
#             raise FileNotFoundError("Model weight file not found")
#         # Extract the output of the feature extraction layer
#         self.feature_extractor = create_feature_extractor(model, feature_nodes)
#         # Select the specified layers as the feature extraction layer
#         self.feature_extractor_nodes = feature_nodes
#         self.feature_weights = feature_weights
#         # input normalization
#         self.normalize = transforms.Normalize(feature_normalize_mean, feature_normalize_std)
#         # Freeze model parameters without derivatives
#         for model_parameters in self.feature_extractor.parameters():
#             model_parameters.requires_grad = False
#         self.use_huber = use_huber
#         self.feature_extractor.eval()

#     def forward(self, sr_tensor: Tensor, gt_tensor: Tensor):
#         assert sr_tensor.size() == gt_tensor.size(), "Two tensor must have the same size"
#         device = sr_tensor.device

#         losses = []
#         # input normalization
#         sr_tensor = self.normalize(sr_tensor)
#         gt_tensor = self.normalize(gt_tensor)

#         # Get the output of the feature extraction layer
#         sr_feature = self.feature_extractor(sr_tensor)
#         gt_feature = self.feature_extractor(gt_tensor)

#         # Compute feature loss
#         for i in range(len(self.feature_extractor_nodes)):
#             if not self.use_huber:
#                 losses.append(self.feature_weights[i]*F_torch.l1_loss(sr_feature[self.feature_extractor_nodes[i]],
#                                             gt_feature[self.feature_extractor_nodes[i]]))
#             else:
#                 losses.append(self.feature_weights[i]*F_torch.huber_loss(sr_feature[self.feature_extractor_nodes[i]],
#                                             gt_feature[self.feature_extractor_nodes[i]], delta=2))
#         losses = torch.Tensor([losses]).to(device)

#         return losses
