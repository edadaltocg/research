def get_lr(it: int, lr_decay_iters: int, learning_rate: float, warmup_iters: int, min_lr: float):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


linear_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=150)
cosine_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total)
schedulers = [linear_warmup, cosine_decay]
scheduler = torch.optim.lr_scheduler.ChainedScheduler(schedulers)
