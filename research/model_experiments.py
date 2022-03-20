# %%
import torch
import torchvision.models as models
import numpy as np
from torchvision import transforms
from PIL import Image
# %%
resnet18 = models.resnet18(pretrained=True)

# %%
class ReductionResnet18():
    def __init__(self):
        pass
    
    def forward(self):
        pass
# %%
layer_counter = 0
for n,c in resnet18.named_children():
    print("Children Counter: ",layer_counter," Layer Name: ",n,)
    layer_counter+=1

# %% 
print(resnet18)   
#%%
# from torchvision.models.feature_extraction import get_graph_node_names

# train_nodes, eval_nodes = get_graph_node_names(resnet18)

# print(train_nodes)

# %%
from torchvision.models.feature_extraction import create_feature_extractor

return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    'layer4': 'layer4',
}
feat_ext = create_feature_extractor(resnet18, return_nodes=return_nodes)

# %%
print(return_nodes)
print(feat_ext)
# %%
# %%
feat_ext2 = create_feature_extractor(resnet18,return_nodes={f'layer{k}': str(v)for v, k in enumerate([1, 2, 3, 4])})
print(return_nodes)
print(feat_ext2)

# %%
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

class Resnet18Self(torch.nn.Module):
    def __init__(self):
        super(Resnet18Self, self).__init__()
        # Get a resnet50 backbone
        m = models.resnet18()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})

        self.body.eval()
        # Dry run to get number of channels for FPN
        # inp = torch.randn(2, 3, 224, 224)
        # with torch.no_grad():
        #     out = self.body(inp)
        # in_channels_list = [o.shape[1] for o in out.values()]
        # # Build FPN
        # self.out_channels = 256
        # self.fpn = FeaturePyramidNetwork(
        #     in_channels_list, out_channels=self.out_channels,
        #     extra_blocks=LastLevelMaxPool())
        
        # Build FPN
        
        
    def forward(self, x):
        features = self.body(x)
        # embedding = torch.ops.mynamespace.reduction(
        #     features.get("layer1"),
        #     features.get("layer2"),
        #     features.get("layer3"),
        #     features.get("layer4"),
        # )
        return features

# %%
import torchvision
import torchextractor as tx
from torch.onnx import register_custom_op_symbolic


# def register_custom_op() -> None:
#     torch.ops.load_library(
#         "/home/saktiman/Dev-ai/image_similarity/scripts/build/lib.linux-x86_64-3.9/reduction_op.cpython-39-x86_64-linux-gnu.so"
#     )
#     """register_custom_op method registers the custom reduction operation on torch.onnx"""

#     def my_reuduction(
#         g: object,
#         layer_one: torch.Tensor,
#         layer_two: torch.Tensor,
#         layer_three: torch.Tensor,
#         layer_four: torch.Tensor,
#     ) -> object:
#         return g.op(
#             "mydomain::reduction", layer_one, layer_two, layer_three, layer_four
#         )

#     register_custom_op_symbolic("mynamespace::reduction", my_reuduction, 9)



class ReductionResNet(torch.nn.Module):
    """Custom reduction res net for get calculating embeddings for the
    Args:
        torch ([type]): [description]
    """

    def __init__(self) -> None:
        """method setup all the necessary variable and object to create a model."""

        super(ReductionResNet, self).__init__()
        # register_custom_op_symbolic()
        self.base_model = torchvision.models.resnet18(pretrained=True)
        self.model = tx.Extractor(
            self.base_model, ["layer1", "layer2", "layer3", "layer4"]
        )
        self.model.eval()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """forward methods takes a input tensor as [B, C, H, W] pass it
        throught the model and returns the generated embedding
        Args:
            input (torch.Tensor):
        Returns:
            embedding: 512 size embedding consisting of image features.
        """
        _, features = self.model(input)
        # embedding = torch.ops.mynamespace.reduction(
        #     features.get("layer1"),
        #     features.get("layer2"),
        #     features.get("layer3"),
        #     features.get("layer4"),
        # )
        return features

# %%
input1 = torch.randn(1, 3, 416, 416)
redresnet = ReductionResNet()
redrs = redresnet.forward(input1)
# %%
print(redrs['layer1'].shape)
# %%
print(redrs['layer2'].shape)
# %%
print(redrs['layer3'].shape)
# %%
print(redrs['layer4'].shape)
# %%
input2 = torch.randn(1, 3, 416, 416)
resnetself = Resnet18Self()
rs = resnetself.forward(input2)

print(rs['0'].shape)

# %%
print(rs['1'].shape)
# %%
print(rs['2'].shape)
# %%
print(rs['3'].shape)
# %%
# %%
input3 = torch.randn(1, 3, 416, 416)
rs18 = resnet18(input3)
print(rs18)
# %%
print(resnet18)
# %%
print(resnetself)
# %%
print(redresnet)
# %%
