"""
Batch=1 Prefill Benchmark Script
测试batch=1的prefill性能，支持4096-8192序列长度，causal和非causal模式
默认参数: head_dim=128, qo_heads=32, kv_heads=2
只测试FA2后端
"""

import torch
import triton
import flashinfer


def benchmark_prefill(seq_len, causal=True, head_dim=128, qo_heads=32, kv_heads=2):
    """
    测试单个prefill配置的性能
    """
    print(f"测试配置: seq_len={seq_len}, causal={causal}, head_dim={head_dim}, qo_heads={qo_heads}, kv_heads={kv_heads}")
    
    # 创建输入张量
    q = torch.randn(seq_len, qo_heads, head_dim, dtype=torch.half, device="cuda")
    k = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
    v = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
    
    # 只测试FA2后端
    try:
        ms_fa2 = triton.testing.do_bench(
            lambda: flashinfer.single_prefill_with_kv_cache(
                q, k, v, causal=causal, backend="fa2"
            ),
            warmup=50,
            rep=500,
        )
        
        # 计算FLOPS (简化计算)
        flops_fa2 = seq_len * seq_len * qo_heads * head_dim * 4 / ms_fa2 / 1e9
        if causal:
            flops_fa2 /= 2  # causal大约减半计算量
            
        print(f"  FA2: {ms_fa2:.3f}ms, {flops_fa2:.3f} TFLOPs/s")
        
    except Exception as e:
        print(f"  FA2失败: {e}")
    
    print()


def main():
    """主函数：运行所有测试"""
    print("=== Batch=1 Prefill Benchmark (FA2 Only) ===")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"计算能力: {torch.cuda.get_device_capability()}")
    print()
    
    # 默认参数
    head_dim = 128
    qo_heads = 32
    kv_heads = 2
    
    # 测试序列长度
    seq_lengths = [4096, 5120, 6144, 7168, 8192]
    
    # 测试Causal模式
    print("=== Causal Mode ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=True, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads)
    
    # 测试Non-Causal模式
    print("=== Non-Causal Mode ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=False, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        exit(1)
    
    main() 