import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.optim.lr_scheduler
import torch.utils.data
from torch.profiler import ProfilerActivity
from torch.utils.data import DataLoader
from tqdm import tqdm

from research.nn.layers.utils import init_weights
from research.trainer._base import TrainerBase
from research.utils import move_to_device

log = logging.getLogger(__file__)


@dataclass
class TrainerSingleDevice(TrainerBase):
    device: str | torch.device = "cpu"

    def _init_training_state(self):
        pass

    def _setup_device(self):
        self.device = torch.device(self.device)

    def _setup_model(self):
        self.model = self.model.to_empty(device=self.device)
        self.model = self.model.apply(lambda x: init_weights(x, self.init_weights))
        if self.compile_model:
            self.model = torch.compile(self.model)
        log.info("Model setup done")

    def _train_one_step(self, batch):
        self.model.train()
        inputs, targets = move_to_device(batch, self.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss = loss / self.accumulation_steps
        loss.backward()

        return loss * self.accumulation_steps

    def train(self):
        if not self.optimizer or not self.criterion:
            return
        if not self.metrics:
            self.metrics = defaultdict(list)
        epbar = tqdm(total=self.num_epochs - self._epoch, unit="epoch", position=0, colour="green")
        for epoch in range(self._epoch, self.num_epochs):
            self.epoch = epoch

            if self.train_loader:
                bpbar = tqdm(total=len(self.train_loader), unit="batch", position=1, colour="blue")
                for batch_idx, batch in enumerate(self.train_loader):
                    self._global_step += 1
                    loss = self._train_one_step(batch)

                    if torch.any(torch.isnan(loss)):
                        log.error("Loss is nan")
                        sys.exit(1)

                    if (batch_idx + 1) % self.accumulation_steps == 0 or batch_idx == len(self.train_loader) - 1:
                        if self.gradient_clip_val is not None:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

                        log.debug("Optimizer step")
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if self._global_step % self.log_interval == 0:
                        current_time = time.time()
                        time_elapsed = current_time - self.last_time
                        self.last_time = current_time
                        throughput = self.batch_size * self.log_interval / time_elapsed  # batches per second
                        lr = self.optimizer.param_groups[0]["lr"]

                        self.metrics["lr"].append(lr)
                        self.metrics["train/throughput"].append(throughput)
                        self.metrics["train/loss"].append(loss.item())
                        self.train_logger.log(self.metrics, self._global_step)
                        bpbar.set_postfix({m: v[-1] for m, v in self.metrics.items()})
                        bpbar.update(self.log_interval)

                    if self._global_step % self.eval_interval == 0 and self.val_loader is not None:
                        t0 = time.time()
                        avg_loss = self.evaluate()
                        t1 = time.time()
                        throughput = self.batch_size * len(self.val_loader) / (t1 - t0 + 1e-8)
                        self.metrics["val/throughput"].append(throughput)
                        self.metrics["val/loss"].append(avg_loss)
                        if avg_loss < self.best_val_loss:
                            self.best_val_loss = avg_loss
                            self.save_checkpoint(best=True)
                        self.train_logger.log(self.metrics, self._global_step)

                    if self._global_step % self.save_interval == 0:
                        self.save_checkpoint()

                bpbar.close()

                if self.lr_scheduler:
                    self.lr_scheduler.step()  # epoch based LR scheduler

            epbar.set_postfix({m: v[-1] for m, v in self.metrics.items()})
            epbar.update(1)
        epbar.close()

        self.save_checkpoint(last=True)
        self.export_model()

    def _evaluate_one_step(self, batch):
        inputs, targets = move_to_device(batch, self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

        return loss

    def evaluate(self):
        if not self.val_loader:
            raise

        self.model.eval()
        total_loss = torch.zeros(1, device=self.device)
        num_batches = 0

        pbar = tqdm(total=len(self.val_loader), unit="val_batch", position=2, colour="red")
        for batch_idx, batch in enumerate(self.val_loader):
            loss = self._evaluate_one_step(batch)
            total_loss += loss
            num_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.update(self.log_interval)

        avg_loss = total_loss / num_batches
        return avg_loss

    def load_checkpoint(self):
        if self.checkpoint_path:
            checkpoint_path = self.checkpoint_path
        elif self.resume_from_checkpoint:
            checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_step_*.ckpt"))
            if not checkpoint_files:
                log.info(f"No checkpoints found in {self.checkpoint_dir} to resume from. Starting from scratch.")
                return
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split("_step_")[-1]))
            checkpoint_path = latest_checkpoint
        else:
            return  # do not load checkpoint

        log.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self._global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics = checkpoint.get("metrics", defaultdict(list))  # load metrics if available

    def save_checkpoint(self, checkpoint_path=None, best=False, last=False):
        if not checkpoint_path:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_step_{self._global_step}.ckpt"
        if best:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_best.ckpt"
        if last:
            checkpoint_path = self.checkpoint_dir / f"{self.checkpoint_prefix}_last.ckpt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            "global_step": self._global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "metrics": self.metrics,
            "rng": {
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        log.info(f"Checkpoint saved to: {checkpoint_path}")

        # Clean up old checkpoints
        checkpoint_files = list(self.checkpoint_dir.glob(f"{self.checkpoint_prefix}_step_*.ckpt"))
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_step_")[-1]), reverse=True)
        for file in checkpoint_files[self.num_checkpoints :]:
            os.remove(file)
        log.info(f"Cleaned up old checkpoints, keeping last {self.num_checkpoints} checkpoints.")

    def export_model(self, name="model.pt"):
        torch.save(self.model.state_dict(), self.output_dir / name)
        log.info(f"Exported model to: {self.output_dir / name}")

    def tune(self):
        max_batch_size = self.batch_size
        log.info("Starting batch size tuning...")
        while True:
            try:
                test_loader = DataLoader(
                    self.train_dataset,
                    batch_size=max_batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                )
                batch = next(iter(test_loader))
                inputs = move_to_device(batch, self.device)
                targets = move_to_device(batch, self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                torch.cuda.empty_cache()  # this line will have no effect on CPU, just for code consistency
                max_batch_size *= 2
                self.train_logger.logger.info(f"Trying batch size: {max_batch_size}")
            except RuntimeError as e:
                if "out of memory" in e:  # no effect on CPU
                    max_batch_size = max_batch_size // 2
                    self.train_logger.logger.info(
                        f"Out of memory (CPU RAM) at batch size {max_batch_size * 2}, reducing to {max_batch_size}"
                    )
                    break
                else:
                    raise e

        self.batch_size = max_batch_size

        best_workers = 0
        best_throughput = 0
        cpu_count = os.cpu_count() or 4
        for num_workers in range(1, min(cpu_count, 16) + 1):
            test_loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                num_workers=num_workers,
                pin_memory=self.pin_memory,
            )

            start_time = time.time()
            batches_processed = 0
            for batch in test_loader:
                inputs = move_to_device(batch, self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)  # removed amp for CPU tune
                batches_processed += 1
                if batches_processed >= 10:
                    break
            end_time = time.time()
            throughput = batches_processed / (end_time - start_time)

            if throughput > best_throughput:
                best_throughput = throughput
                best_workers = num_workers

        self.train_logger.logger.info(f"Optimal number of workers: {best_workers}")
        self.num_workers = best_workers
        self._setup_dataloaders()  # recreate dataloaders with new num_workers

    def setup_profiler(self):
        profiling_schedule = torch.profiler.schedule(wait=1, warmup=5, active=3, repeat=1)
        on_trace_ready = torch.profiler.tensorboard_trace_handler(str(self.log_dir / "tb_trace"))
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]
        torch_profiler = torch.profiler.profile(
            activities=activities,
            profile_memory=True,
            schedule=profiling_schedule,
            on_trace_ready=on_trace_ready,
        )
        return torch_profiler

    def profile(self):
        if not self.optimizer or not self.criterion:
            return

        pbar = tqdm(total=10, unit="epoch", position=0, colour="magenta")
        prof = self.setup_profiler()
        prof.start()
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx > 10:  # Profile for a limited number of steps
                break
            inputs, targets = move_to_device(batch, self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            pbar.update()
            prof.step()
        prof.stop()
        log.info("\n" + prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))
        # prof.export_chrome_trace(str(self.log_dir / "profiler_trace.json"))


class TrainerCPU(TrainerSingleDevice):
    device = "cpu"


class TrainerGPU(TrainerSingleDevice):
    device = "cuda"


class TrainerMPS(TrainerSingleDevice):
    device = "mps"
