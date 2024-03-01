import logging
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.profiler import ProfilerActivity, schedule
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dest_root = "output/tutorials/fsdp"
dest_root = Path(dest_root)
dest_root.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=dest_root / "train-logs", comment="train")
writer2 = SummaryWriter(log_dir=dest_root / "hparam-logs", comment="hparams")

log = logging.getLogger(__name__)

# setup distributed environment
try:
    dist.init_process_group(backend="nccl", init_method="env://")
    RANK = dist.get_rank()
    WORLD_SIZE = dist.get_world_size()
    device = torch.device(f"cuda:{RANK}")
except ValueError:
    RANK = 0
    WORLD_SIZE = 1 if torch.cuda.is_available() else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{RANK=}, {WORLD_SIZE=}, {device=}")
# device = torch.device("cpu")
logging.basicConfig(level=logging.INFO if RANK == 0 else logging.ERROR)
seed = 42
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# config
seq_len = 2048
vocab_size = 32000
num_samples = 100000
batch_size = 20
step_batch_size = batch_size
num_workers = 8 if WORLD_SIZE >= 1 else 0

d_model = 768
nhead = 8
num_layers = 8
dim_feedforward = d_model * 4
bias = False
activation = F.gelu
grad_clip_norm = 0.1
desired_batch_size = 1024
gradient_accumulation_steps = desired_batch_size // step_batch_size
hparams = {
    k: v
    for k, v in locals().items()
    if k != "writer" and type(v) in [int, float, str, bool, torch.Tensor]
}
toks_multiplier = seq_len * batch_size


# data
dataset = []
for i in range(num_samples):
    start = random.randint(0, vocab_size)
    if start + seq_len + 1 >= vocab_size:
        elem1 = torch.arange(start, vocab_size)
        elem2 = torch.arange(0, start + seq_len + 1 - vocab_size)
        elem = torch.cat([elem1, elem2])
    else:
        elem = torch.arange(start, start + seq_len + 1)
    assert elem.shape == (seq_len + 1,)
    dataset.append(elem.unsqueeze(0))
dataset = torch.utils.data.TensorDataset(torch.concatenate(dataset))
log.info(dataset[:5][0])

if WORLD_SIZE > 1:
    sampler = torch.utils.data.DistributedSampler(dataset, drop_last=True)
else:
    sampler = None
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True,
)
dataloader = iter(dataloader)


# model
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        pos_encoding = torch.randn(1, seq_len, d_model, device=device)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=0.1,
            batch_first=True,
            bias=bias,
        )
        encoder_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x, mask):
        x = self.embedding(x) + self.pos_encoding
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.head(x)
        return x


model = Transformer()
if WORLD_SIZE > 1:
    bf16 = MixedPrecision(
        param_dtype=torch.bfloat16,
        # Gradient comunication precision.
        reduce_dtype=torch.float32,
        # Buffer precision.
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        device_id=RANK,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,  # can increase the training speed in exchange for higher mem
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        mixed_precision=bf16,
        use_orig_params=True,
    )
    compiler_mode = None
    compiler_fullgraph: bool = False
    compiler_backend: str = "inductor"

    model = torch.compile(
        model,
        mode=compiler_mode,
        backend=compiler_backend,
        fullgraph=compiler_fullgraph,
    )
else:
    model = model.to(device)
    # writer.add_graph(model, (x, attn_mask))
log.info(model)
log.info(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

# attn mask added to the attention scores in the self-attention mechanism
# the shape is broadcasted to (batch_size, nhead, tgt_len, src_len)
attn_mask = torch.full((seq_len, seq_len), -float("Inf"), device=device)
attn_mask = torch.triu(attn_mask, diagonal=1)
assert torch.allclose(
    torch.nn.Transformer.generate_square_subsequent_mask(seq_len, device=device),
    attn_mask,
)
log.info(f"{attn_mask=}")

num_steps = len(dataloader) * 1
total = num_steps
optimizer = optim.AdamW(
    model.parameters(), betas=(0.9, 0.95), eps=1e-5, weight_decay=0.1, lr=4e-4
)
linear_warmup = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, total_iters=150
)
cosine_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
schedulers = [linear_warmup, cosine_decay]
scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)
# torch.optim.lr_scheduler.ConstantLR(optimizer, 1)

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

# loop
num_checkpoints = 2
checkpoint_every = max(total // max(1, num_checkpoints), 1)
pbar = tqdm(
    range(total),
    total=total,
    dynamic_ncols=True,
    position=RANK,
    desc=f"Train [{RANK+1}/{WORLD_SIZE}]",
    leave=True,
)
if WORLD_SIZE > 1 and sampler is not None:
    sampler.set_epoch(0)
loss = torch.zeros(2, device=device)
lr_monitor = torch.zeros(1, device=device)
throughout = torch.zeros(2, device=device)
grad_norm = torch.zeros(1, device=device)
peak_gpu_memory = 0
if WORLD_SIZE >= 1:
    torch.cuda.reset_peak_memory_stats(device)
    peak_gpu_memory = torch.cuda.max_memory_allocated(device)
print(f"{RANK=} Peak GPU memory: {peak_gpu_memory / 1024**3:.2f} GB")

device_history = {
    k: torch.zeros((total,), device=device)
    for k in ["train/loss", "lr", "throughput", "mem", "grad_norm"]
}

profiling_schedule = schedule(wait=1, warmup=5, active=3, repeat=1)
on_trace_ready = torch.profiler.tensorboard_trace_handler(str(dest_root / "trace"))
activities = [ProfilerActivity.CPU]
if WORLD_SIZE >= 1:
    activities.append(ProfilerActivity.CUDA)
torch_profiler = torch.profiler.profile(
    activities=activities,
    record_shapes=False,
    profile_memory=False,
    with_stack=True,
    schedule=profiling_schedule,
    on_trace_ready=on_trace_ready,
)
if WORLD_SIZE >= 1:
    torch_profiler.start()

for i in pbar:
    batch = next(dataloader)
    t0_batch = time.time()

    if WORLD_SIZE >= 1:
        torch_profiler.step()
    batch = batch[0].to(device, non_blocking=True)
    inside_loop_batch_size = batch.size(0)
    assert inside_loop_batch_size == batch_size
    x = batch[:, :-1]
    y = batch[:, 1:].clone().to(torch.int64)
    model.train()
    optimizer.zero_grad()
    pred = model.forward(x, mask=attn_mask)
    pred = pred.view(-1, pred.size(-1))
    y = y.view(-1)
    log.debug(f"{pred.shape=}, {y.shape=}")
    device_loss = F.cross_entropy(pred, y, reduction="sum")
    device_loss = device_loss / gradient_accumulation_steps
    device_loss.backward()
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if (i + 1) % gradient_accumulation_steps == 0:
        log.info(f"Step {i+1} optimizer.step()")
        optimizer.step()
        scheduler.step()

    loss[0] += device_loss
    loss[1] += inside_loop_batch_size
    lr_monitor[0] = optimizer.param_groups[0]["lr"]
    throughout[0] += inside_loop_batch_size
    throughout[1] += time.time() - t0_batch
    if WORLD_SIZE >= 1:
        peak_gpu_memory = torch.cuda.max_memory_allocated(device)
    device_history["train/loss"][i] = loss[0] / loss[1]
    device_history["lr"][i] = lr_monitor[0]
    device_history["throughput"][i] = (
        throughout[0] / throughout[1]
    ).item() * toks_multiplier
    device_history["mem"][i] = peak_gpu_memory / 1024**3
    grad_norm = torch.zeros(1, device=device)
    parameters = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        grad_norm += param_norm**2
    total_norm = grad_norm ** (1.0 / 2)
    device_history["grad_norm"][i] = total_norm

    # checkpoint
    if (i + 1) % checkpoint_every == 0:
        if WORLD_SIZE > 1:
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
            ):
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                scheduler_state_dict = scheduler.state_dict()
        else:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
        checkpoint = {
            "model": model_state_dict,
            "optimizer": optimizer_state_dict,
            "scheduler": scheduler_state_dict,
        }
        if RANK == 0:
            print(f"{RANK=} Checkpointing at {i+1=}")
            torch.save(checkpoint, dest_root / f"checkpoint-{i+1}.pt")

    pbar.update(1)
    latest_history = {k: v[i].item() for k, v in device_history.items()}
    pbar.set_postfix(latest_history)
    if RANK <= 1:
        for k, v in latest_history.items():
            writer.add_scalar(f"{k}/{RANK}", v, i)

    # eval
    # for batch in eval_dataloader:
    #     batch = batch[0].to(device, non_blocking=True)
    #     x = batch[:, :-1]
    #     y = batch[:, 1:].clone().to(torch.int64)
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model.forward(x, mask=attn_mask)
    #     pred = pred.view(-1, pred.size(-1))
    #     y = y.view(-1)
    #     eval_device_loss = F.cross_entropy(pred, y, reduction="sum")

if WORLD_SIZE >= 1:
    torch_profiler.stop()
pbar.close()
if WORLD_SIZE > 1:
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(throughout, op=dist.ReduceOp.SUM)
    for k in device_history:
        dist.all_reduce(device_history[k], op=dist.ReduceOp.AVG)

    dist.barrier()
epoch_loss = loss[0] / loss[1]
epoch_loss = epoch_loss.item()
log.info(f"{epoch_loss=}")

latest_history = {f"hparams/{k}": v[-1].item() for k, v in device_history.items()}
log.info(f"{latest_history=}")
if RANK == 0:
    writer2.add_hparams(hparams, latest_history, run_name="last")
writer.close()

if WORLD_SIZE > 1:
    dist.barrier()
    dist.destroy_process_group()

"""
torchrun --standalone --nnodes=1 --nproc_per_node=2 tutorials/fsdp.py
"""
