import torch.distributed as dist
import torch.utils.data
from torch.utils.data.distributed import DistributedSampler

import utils


def test_step_accumulation():
    """
    Rank: 1, Step: tensor([1.], device='cuda:1'), Loss: 1.0
    Rank: 1, Step: tensor([2.], device='cuda:1'), Loss: 2.0
    Rank: 1, Step: tensor([3.], device='cuda:1'), Loss: 3.0
    Rank: 1, Step: tensor([4.], device='cuda:1'), Loss: 4.0
    Rank: 0, Step: tensor([1.], device='cuda:0'), Loss: 1.0
    Rank: 0, Step: tensor([2.], device='cuda:0'), Loss: 2.0
    Rank: 1, Step: tensor([5.], device='cuda:1'), Loss: 5.0
    Before barrier: Rank: 1, Step: tensor([5.], device='cuda:1'), Loss: 5.0
    Rank: 0, Step: tensor([3.], device='cuda:0'), Loss: 3.0
    Rank: 0, Step: tensor([4.], device='cuda:0'), Loss: 4.0
    Rank: 0, Step: tensor([5.], device='cuda:0'), Loss: 5.0
    Before barrier: Rank: 0, Step: tensor([5.], device='cuda:0'), Loss: 5.0
    After barrier: Rank: 0, Step: tensor([5.], device='cuda:0'), Loss: 10.0
    After barrier: Rank: 1, Step: tensor([5.], device='cuda:1'), Loss: 5.0
    """
    utils.distrib_setup()
    utils.seed_all(42)
    rank = utils.get_rank()
    world_size = utils.get_world_size()

    w = torch.randn(10, 1024**2, device=rank)
    dataset = torch.utils.data.TensorDataset(torch.randn(1000, 1024**2))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=100,
        pin_memory=True,
        num_workers=2,
        shuffle=False,
    )
    torch.cuda.set_device(rank)
    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    step = torch.zeros(1, device=rank)
    step_live = torch.zeros(1, device=rank)
    one = torch.ones(1, device=rank)
    loss = torch.zeros(1, device=rank)
    init_start_event.record(torch.cuda.current_stream())
    for x in dataloader:
        x = w.mm(x[0].to(rank).t())
        step += one
        loss += one
        print(f"Rank: {rank}, Step: {step}, Loss: {loss.item()}")
        # force sync (bad)
        # step_live += one
        # dist.reduce(step_live, 0, op=dist.ReduceOp.SUM)
    init_end_event.record(torch.cuda.current_stream())
    print(
        "Elapsed time in seconds: ",
        init_start_event.elapsed_time(init_end_event) / 1000,
    )
    print(f"Before barrier: Rank: {rank}, Step: {step}, Loss: {loss.item()}")
    dist.reduce(loss, 0, op=dist.ReduceOp.SUM)
    dist.reduce(step, 0, op=dist.ReduceOp.SUM)
    print(f"After barrier: Rank: {rank}, Step: {step}, Loss: {loss.item()}")
    if utils.is_main_process():
        assert step.cpu().item() == 10
        assert loss.cpu().item() == 10.0
    utils.distrib_cleanup()


if __name__ == "__main__":
    test_step_accumulation()
