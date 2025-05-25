"""
Batch=1 Prefill Benchmark Script
测试batch=1的prefill性能，支持4096-8192序列长度，causal和非causal模式
默认参数: head_dim=128, qo_heads=32, kv_heads=2
只测试FA2后端，现在包含fp16和kv fp8测试
"""

import torch
import triton
import flashinfer


def benchmark_prefill(seq_len, causal=True, head_dim=128, qo_heads=32, kv_heads=2, test_fp8=False):
    """
    测试单个prefill配置的性能
    """
    dtype_str = "FP8" if test_fp8 else "FP16"
    print(f"测试配置: seq_len={seq_len}, causal={causal}, head_dim={head_dim}, qo_heads={qo_heads}, kv_heads={kv_heads}, dtype={dtype_str}")
    
    if test_fp8:
        # FP8 KV cache测试
        q = torch.randn(seq_len, qo_heads, head_dim, dtype=torch.half, device="cuda")
        # 创建fp16的k, v，然后转换为fp8
        k_fp16 = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
        v_fp16 = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
        
        # 计算缩放因子 (简化的量化方案)
        k_scale = k_fp16.abs().max().item() / 240.0  # 留一些headroom
        v_scale = v_fp16.abs().max().item() / 240.0
        
        # 转换为fp8
        k = (k_fp16 / k_scale).to(torch.float8_e4m3fn)
        v = (v_fp16 / v_scale).to(torch.float8_e4m3fn)

        scale_k = torch.tensor(k_scale, dtype=torch.float32, device="cuda") 
        scale_v = torch.tensor(v_scale, dtype=torch.float32, device="cuda")
        
        try:
            ms_fa2 = triton.testing.do_bench(
                lambda: flashinfer.single_prefill_with_kv_cache(
                    q, k, v, causal=causal, backend="fa2",
                    scale_k=scale_k,
                    scale_v=scale_v
                ),
                warmup=5,
                rep=50,
            )
            
            # 计算FLOPS (简化计算)
            flops_fa2 = seq_len * seq_len * qo_heads * head_dim * 4 / ms_fa2 / 1e9
            if causal:
                flops_fa2 /= 2  # causal大约减半计算量
                
            print(f"  FA2 (KV FP8): {ms_fa2:.3f}ms, {flops_fa2:.3f} TFLOPs/s")
            
        except Exception as e:
            print(f"  FA2 (KV FP8)失败: {e}")
    else:
        # FP16测试
        q = torch.randn(seq_len, qo_heads, head_dim, dtype=torch.half, device="cuda")
        k = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
        v = torch.randn(seq_len, kv_heads, head_dim, dtype=torch.half, device="cuda")
        
        try:
            ms_fa2 = triton.testing.do_bench(
                lambda: flashinfer.single_prefill_with_kv_cache(
                    q, k, v, causal=causal, backend="fa2"
                ),
                warmup=5,
                rep=50,
            )
            
            # 计算FLOPS (简化计算)
            flops_fa2 = seq_len * seq_len * qo_heads * head_dim * 4 / ms_fa2 / 1e9
            if causal:
                flops_fa2 /= 2  # causal大约减半计算量
                
            print(f"  FA2 (FP16): {ms_fa2:.3f}ms, {flops_fa2:.3f} TFLOPs/s")
            
        except Exception as e:
            print(f"  FA2 (FP16)失败: {e}")
    
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
    seq_lengths = [4096, 6144, 8192, 16384]
    
    # 测试Causal模式 - FP16
    print("=== Causal Mode - FP16 ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=True, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads, test_fp8=False)
    
    # 测试Non-Causal模式 - FP16
    print("=== Non-Causal Mode - FP16 ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=False, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads, test_fp8=False)
    
    # 测试Causal模式 - KV FP8
    print("=== Causal Mode - KV FP8 ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=True, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads, test_fp8=True)
    
    # 测试Non-Causal模式 - KV FP8
    print("=== Non-Causal Mode - KV FP8 ===")
    for seq_len in seq_lengths:
        benchmark_prefill(seq_len, causal=False, head_dim=head_dim, qo_heads=qo_heads, kv_heads=kv_heads, test_fp8=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("错误: 需要CUDA支持")
        exit(1)
    
    main() 