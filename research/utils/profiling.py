"""
GPU operations can be break down into:
    1. COMP - Computatiom (They are responsible for all of the number-crunching necessary for model execution.)
    2. MEM - Memory (H2D, D2H, D2D stands for Host to Device, Device to Host and Device to Device respectively)
    3. COMM - Communication (NCCL_AllGather, NCCL_ReduceScatter, NCCL_AllReduce)

Temporal breakdown of GPU operations:
    1. Idle time
    2. Compute time
    3. Non compute time

trace files can easily run into hundreds of MBs

https://pytorch.org/blog/trace-analysis-for-masses/
https://hta.readthedocs.io/en/latest/
https://pytorch.org/docs/master/profiler.html
"""

import os
import shutil

import genai.distrib as distrib
import torch
import torch.profiler
from hta.trace_analysis import TraceAnalysis
from torch.profiler import schedule, tensorboard_trace_handler
from tqdm import tqdm


def profile():
    utils.distrib_setup()

    trace_path = "traces/test"
    # clear folder
    if utils.is_main_process():
        if os.path.exists(trace_path):
            shutil.rmtree(trace_path)

    tracing_schedule = schedule(skip_first=1, wait=1, warmup=1, active=2, repeat=1)
    trace_handler = tensorboard_trace_handler(dir_name=trace_path, use_gzip=True)
    with torch.profiler.profile(
        schedule=tracing_schedule,
        on_trace_ready=trace_handler,
        profile_memory=True,
        record_shapes=True,
        with_stack=True,
        with_flops=True,
    ) as p:
        for i in tqdm(range(5), disable=not distrib.is_main_process()):
            # function call
            # 1024 ** 3 * 4 * 1 = 4GB
            x = torch.randn(1, 1024, 1024, 1024, dtype=torch.float32)
            x = x.to("mps")
            # device = f"cuda:{distrib.get_local_rank()}"
            # x = x.to(device)
            p.step()

    if utils.is_main_process():
        # `cpu_time`, `cuda_time`, `cpu_time_total`,
        # `cuda_time_total`, `cpu_memory_usage`, `cuda_memory_usage`,
        # `self_cpu_memory_usage`, `self_cuda_memory_usage`, `count`
        print(
            p.key_averages().table(
                sort_by="self_cuda_memory_usage",
                row_limit=-1,
                top_level_events_only=False,
            )
        )
        print(f"Trace files saved to {trace_path}")
        analyzer = TraceAnalysis(trace_dir=trace_path)

        # Temporal breakdown
        temporal_breakdown_df = analyzer.get_temporal_breakdown()
        temporal_breakdown_df.to_csv(os.path.join(trace_path, "temporal_breakdown.csv"))

        # Idle time breakdown
        idle_time_df = analyzer.get_idle_time_breakdown()
        idle_time_df.to_csv(os.path.join(trace_path, "idle_time_breakdown.csv"))

        # Kernel breakdown
        kernel_breakdown_df = analyzer.get_gpu_kernel_breakdown()
        kernel_breakdown_df.to_csv(os.path.join(trace_path, "kernel_breakdown.csv"))

        # Communication computation overlap
        comm_comp_overlap_df = analyzer.get_comm_comp_overlap()
        comm_comp_overlap_df.to_csv(os.path.join(trace_path, "comm_comp_overlap.csv"))

        # Memory bandwidth time series
        memory_bw_series = analyzer.get_memory_bw_time_series()
        memory_bw_series.to_csv(os.path.join(trace_path, "memory_bw_series.csv"))

        # Memory bandwidth summary
        memory_bw_summary = analyzer.get_memory_bw_summary()
        memory_bw_summary.to_csv(os.path.join(trace_path, "memory_bw_summary.csv"))

        # Queue length time series
        ql_series = analyzer.get_queue_length_time_series()
        ql_series.to_csv(os.path.join(trace_path, "ql_series.csv"))

        # Queue length summary
        ql_summary = analyzer.get_queue_length_summary()
        ql_summary.to_csv(os.path.join(trace_path, "ql_summary.csv"))

        # CUDA kernel launch statistics
        cuda_kernel_launch_stats = analyzer.get_cuda_kernel_launch_stats()
        cuda_kernel_launch_stats.to_csv(
            os.path.join(trace_path, "cuda_kernel_launch_stats.csv")
        )

        # Frequent CUDA kernel sequences
        frequent_patterns_df = analyzer.get_frequent_cuda_kernel_sequences(
            operator_name="aten::linear", output_dir=trace_path
        )
        frequent_patterns_df.to_csv(os.path.join(trace_path, "frequent_patterns.csv"))
