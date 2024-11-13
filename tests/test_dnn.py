from copy import deepcopy

import torch
from torch import nn

import dnn.data
import dnn.transformer
import utils

utils.seed_all(42)


def test_layer_norm():
    x = torch.randn(2, 3, 4)

    ln1 = dnn.transformer.LayerNorm(4, eps=1e-5, bias=True)
    ln2 = nn.LayerNorm(4, eps=1e-5, bias=True)

    ln1.reset_parameters()
    ln2.reset_parameters()

    y1 = ln1(x)
    y2 = ln2(x)

    assert y1.size() == (2, 3, 4)
    assert y2.size() == (2, 3, 4)
    assert torch.allclose(ln1.weight, ln2.weight)
    assert torch.allclose(ln1.bias, ln2.bias)
    assert torch.allclose(y1, y2), (y1 - y2).abs().max()


def test_module_list_fx():
    class Encoder(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            l = nn.Linear(10, 10)
            self.encoder = nn.ModuleList([deepcopy(l) for _ in range(2)])

        def forward(self, x):
            for layer in self.encoder:
                x = layer(x)
            return x

    x = torch.randn(2, 10)
    encoder = Encoder()
    print(encoder.state_dict())
    train_nodes, eval_nodes = utils.get_graph_node_names(encoder)
    print("train nodes", train_nodes)
    print("eval nodes", eval_nodes)
    node = train_nodes[-1]
    fe = utils.create_feature_extractor(encoder, [node])
    y = fe(x)
    assert y[node].size() == (2, 10)
