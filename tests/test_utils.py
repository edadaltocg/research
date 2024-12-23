from pathlib import Path

import torch
from torch import nn
from utils.utils import LoadPreTrainedModelWithLowMemoryContext

dim = 1024


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        buffer = nn.Parameter(torch.randn(1, dim))
        self.register_buffer("buffer", buffer)

    def forward(self, x):
        x = self.linear(x) + self.buffer
        x = self.dropout(x)
        x = self.relu(x)
        return x


class ModifiedModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.extra_linear = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.linear(x) + self.buffer
        x = self.dropout(x)
        x = self.relu(x)
        x = self.extra_linear(x)
        return x


model = DummyModel()
state_dict = model.state_dict()
path = Path("output/tests/weights")
path.mkdir(parents=True, exist_ok=True)
torch.save(state_dict, path / "dummy_model_original.pth")


def test_load_pretrained_model_with_low_memory_context():
    with LoadPreTrainedModelWithLowMemoryContext(
        path / "dummy_model_original.pth",
        torch.device("cpu"),
    ) as ctx:
        w = ctx.state_dict
        model = DummyModel()
        assert model.linear.weight.device == torch.device("meta")
        ctx.load_state_dict(model)

        print("children")
        for submodule in model.children():
            print(f"{submodule=}")
        print("-" * 80)

        print("named_children")
        for submodule_name, submodule in model.named_children():
            print(f"{submodule_name=}, {submodule=}")
        print("-" * 80)

        print("state_dict")
        for k, v in model.state_dict().items():
            print(f"{k=}, {v=}, {v.device=}, {v.requires_grad=}")
        print("-" * 80)

        print("named_modules")
        for submodule_name, submodule in model.named_modules():
            print(f"{submodule_name=}, {submodule=}")
            for param_name, param in submodule.named_parameters():
                print(f"{param_name=}, {param=}")
            for buffer_name, buffer in submodule.named_buffers():
                print(f"{buffer_name=}, {buffer=}")
        print("-" * 80)

        print("named_buffers")
        for buffer_name, buffer in model.named_buffers():
            print(f"{buffer_name=}, {buffer=}")
        print("-" * 80)

        print(f"{model.state_dict()=}")
        print(f"{w=}")

        x = torch.randn(1, dim)
        assert x.device == torch.device("meta")

    assert model.linear.weight.device == torch.device("cpu")
    x = torch.randn(1, dim)
    assert x.device == torch.device("cpu")
    model = model.float()
    assert model(x).device == torch.device("cpu")

    assert all([p.device == torch.device("cpu") for p in model.parameters()])
    assert all([p.device == torch.device("cpu") for p in model.buffers()])
    assert all([p.dtype == torch.float32 for p in model.parameters()])
    assert all([p.dtype == torch.float32 for p in model.buffers()])


def test_minimal_load_pretrained_model_with_low_memory_context():
    with LoadPreTrainedModelWithLowMemoryContext(
        path / "dummy_model_original.pth",
        torch.device("cpu"),
        torch.float32,
    ) as ctx:
        w = ctx.state_dict
        model = DummyModel()
        assert model.linear.weight.device == torch.device("meta")
        print(f"{model.state_dict()=}")
        print(f"{w=}")
        ctx.load_state_dict(model)
        print(f"{model.state_dict()=}")
        print(f"{w=}")

    assert all([p.device == torch.device("cpu") for p in model.parameters()])
    assert all([p.device == torch.device("cpu") for p in model.buffers()])
    assert all([p.dtype == torch.float32 for p in model.parameters()])
    assert all([p.dtype == torch.float32 for p in model.buffers()])


def test_modified_model():
    with LoadPreTrainedModelWithLowMemoryContext(
        path / "dummy_model_original.pth",
        torch.device("cpu"),
        torch.float32,
    ) as ctx:
        w = ctx.state_dict
        model = ModifiedModel()
        assert model.linear.weight.device == torch.device("meta")
        print(f"{model.state_dict()=}")
        print(f"{w=}")
        ctx.load_state_dict(model)
        print(f"{model.state_dict()=}")
        print(f"{w=}")

    assert all([p.device == torch.device("cpu") for p in model.parameters()])
    assert all([p.device == torch.device("cpu") for p in model.buffers()])
    assert all([p.dtype == torch.float32 for p in model.parameters()])
    assert all([p.dtype == torch.float32 for p in model.buffers()])
