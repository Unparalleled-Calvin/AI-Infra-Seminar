import torch
from torch.nn import MultiheadAttention
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda')
model = MultiheadAttention(embed_dim=512, num_heads=8).to(device)
x = torch.randn(100, 32, 512).to(device)
key = torch.randn(100, 32, 512).to(device)
value = torch.randn(100, 32, 512).to(device)

with profile(
    activities=[ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/profiler"),
    record_shapes=False,
    with_stack=True,
) as prof:
    for _ in range(3):
        prof.step()
        with record_function("Attension"):
            model(x, key, value)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
