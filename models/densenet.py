import torch.nn as nn
import monai.networks.nets as nets


class AxisDenseNet264(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels):
        super(AxisDenseNet264, self).__init__()
        self.base_model = nets.DenseNet264(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.base_model.class_layers = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(output_size=1),
            nn.Flatten(start_dim=1, end_dim=-1),
        )
        self.score_cls_head = nn.Linear(
            in_features=2688,
            out_features=out_channels,
            bias=True,
        )
        self.axis_cls_head = nn.Linear(in_features=2688, out_features=3, bias=True)

    def forward(self, x):
        x = self.base_model.features(x)
        x = self.base_model.class_layers(x)
        score_cls = self.score_cls_head(x)
        axis_cls = self.axis_cls_head(x)
        return score_cls, axis_cls
