import torch
import torchvision

from utils import create_feature_extractor, get_graph_node_names, seed_all
from vit.model import (
    VisionTransformerEncoder,
    convert_from_pytorch_vit,
    vit_b_16_config,
)

seed_all(42)


def test_convert_from_pytorch_vit():
    config = vit_b_16_config
    model = VisionTransformerEncoder(image_size=224, **config)
    print("My keys", model.state_dict().keys())
    w_pretrained = torch.load("./weights/vit_b_16-c867db91.pth")
    print("Pretrained keys", w_pretrained.keys())
    w_new = convert_from_pytorch_vit(w_pretrained)
    # load the weights
    incompatible_keys = model.load_state_dict(w_new, strict=False)
    print("Incompatible keys", incompatible_keys)
    for k, v in model.state_dict().items():
        assert torch.allclose(v, w_new[k]), f"{k}, {(v - w_new[k]).abs().max()}"

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    y1 = model(x)

    train_nodes, my_eval_nodes = get_graph_node_names(model)

    model = torchvision.models.vision_transformer.vit_b_16()
    model.load_state_dict(w_pretrained)
    model.eval()

    train_nodes, eval_nodes = get_graph_node_names(model)

    print("My nodes", my_eval_nodes)
    print("Torch nodes", eval_nodes)

    fe = create_feature_extractor(model, ["getitem_5"])
    y2 = fe(x)["getitem_5"]
    assert torch.allclose(y1, y2, atol=1e-4), (y1 - y2).abs().max()
