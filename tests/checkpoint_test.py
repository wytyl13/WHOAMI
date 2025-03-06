import pytest
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
def measure_memory():
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    torch.cuda.reset_peak_memory_stats()
    return allocated, max_allocated

def get_dynamic_batch_size(max_batch_size):
    return min(max_batch_size, int(torch.cuda.mem_get_info()[0] / 1e7))


class CheckPointTestModel(nn.Module):
    def __init__(self, depth=10):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1000, 1000),
                nn.GELU(),
                nn.Dropout(0.1)
            ) for _ in range(depth)
        ])
        
    def forward(self, x, use_check_flag):
        for index, layer in enumerate(self.layers):
            if use_check_flag and index % 3 == 0:
                # set the activation checkpoint each three layers.
                with torch.amp.autocast('cuda:0', dtype=torch.float16):
                    # 对检查点区域强制FP16计算
                    x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        return x
                
@pytest.mark.parametrize(
    "use_check_flag, epochs, max_batch_size",
    [
        pytest.param(
            False, 5, 512
        ),
        pytest.param(
            True, 5, 512
        ),
        pytest.param(
            False, 5, 1024
        ),
        pytest.param(
            True, 5, 1024
        )
    ]
)
def test_checkpoint(
    use_check_flag,
    epochs,
    max_batch_size
):
    MEM_GROWTH_THRESHOLD = 0.5
    dynamic_batch_size = get_dynamic_batch_size(max_batch_size)
    print(f"dynamic_batch_size: {dynamic_batch_size}")
    x = torch.randn(dynamic_batch_size, 1000, device=device)
    target = torch.randn(dynamic_batch_size, 1000, device=device)
    model = CheckPointTestModel(depth=300)
    model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    torch.cuda.empty_cache()
    base_alloc, _ = measure_memory()
    
    mem_records = []
    current_alloc = 0
    index = 0
    print(f"\nCheckpoint {use_check_flag} Memory Profile:")
    for epoch in range(epochs):
        index += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda:0'):
            # 自动使用FP16计算模式降低显存需求
            output = model(x, use_check_flag)
            loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        # 显存记录
        current_alloc_ = (torch.cuda.memory_allocated() - base_alloc)/1024**2
        peak_alloc = (torch.cuda.max_memory_allocated() - base_alloc)/1024**2
        mem_records.append((current_alloc_, peak_alloc))
        growth_rate = (current_alloc_ - current_alloc) / current_alloc if current_alloc != 0 else 0
        assert growth_rate < MEM_GROWTH_THRESHOLD, f"Memory leak detected: {growth_rate*100:.2f}% growth at epoch {index}"
        print(f"Epoch {index}: Current={current_alloc_:.2f}MB, Peak={peak_alloc:.2f}MB")
        current_alloc = current_alloc_
        # 缓存清理
        # 显式删除中间变量并调用empty_cache()
        del output, loss
        if (epoch+1) % 5 == 0:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
