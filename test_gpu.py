#!/usr/bin/env python
"""
Quick GPU test script to verify PyTorch can use the GPUs despite compatibility warnings.
"""

import torch
import time

def test_gpu_basic():
    """Test basic GPU operations"""
    print("=" * 60)
    print("GPU Basic Operations Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        return False
    
    print(f"✓ CUDA available")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    print()
    return True


def test_tensor_operations():
    """Test tensor operations on GPU"""
    print("=" * 60)
    print("Testing Tensor Operations on GPU")
    print("=" * 60)
    
    try:
        device = torch.device('cuda:0')
        
        # Create tensors
        print("Creating tensors on GPU...")
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # Matrix multiplication
        print("Performing matrix multiplication...")
        start = time.time()
        z = torch.mm(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Matrix multiplication successful")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Result shape: {z.shape}")
        
        # Convolution (common in neural networks)
        print("\nTesting convolution operation...")
        conv_input = torch.randn(8, 3, 224, 224, device=device)
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        
        start = time.time()
        output = conv(conv_input)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Convolution successful")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during tensor operations: {e}")
        return False


def test_transformer_operations():
    """Test transformer-style operations (relevant for ACT)"""
    print("\n" + "=" * 60)
    print("Testing Transformer Operations (ACT-relevant)")
    print("=" * 60)
    
    try:
        device = torch.device('cuda:0')
        
        # Multi-head attention
        print("Testing multi-head attention...")
        batch_size = 4
        seq_len = 100
        d_model = 256
        nhead = 8
        
        # Create attention layer
        attention = torch.nn.MultiheadAttention(d_model, nhead, batch_first=True).to(device)
        
        # Create input
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        start = time.time()
        output, _ = attention(x, x, x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Multi-head attention successful")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Output shape: {output.shape}")
        
        # Transformer encoder layer
        print("\nTesting transformer encoder layer...")
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            batch_first=True
        ).to(device)
        
        start = time.time()
        output = encoder_layer(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"✓ Transformer encoder layer successful")
        print(f"  Time: {elapsed*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during transformer operations: {e}")
        return False


def test_memory():
    """Test GPU memory allocation"""
    print("\n" + "=" * 60)
    print("GPU Memory Test")
    print("=" * 60)
    
    try:
        device = torch.device('cuda:0')
        
        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        initial_mem = torch.cuda.memory_allocated(device) / 1024**3
        
        print(f"Initial memory: {initial_mem:.2f} GB")
        
        # Allocate some memory
        print("Allocating 1GB tensor...")
        large_tensor = torch.randn(256, 1024, 1024, device=device)
        
        allocated_mem = torch.cuda.memory_allocated(device) / 1024**3
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3
        
        print(f"✓ Allocation successful")
        print(f"  Allocated memory: {allocated_mem:.2f} GB")
        print(f"  Peak memory: {peak_mem:.2f} GB")
        
        # Free memory
        del large_tensor
        torch.cuda.empty_cache()
        
        final_mem = torch.cuda.memory_allocated(device) / 1024**3
        print(f"  Memory after cleanup: {final_mem:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during memory test: {e}")
        return False


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "GPU Functionality Test" + " " * 21 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        "Basic GPU": test_gpu_basic(),
        "Tensor Operations": test_tensor_operations(),
        "Transformer Operations": test_transformer_operations(),
        "Memory Management": test_memory(),
    }
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n🎉 All tests passed! GPU is working correctly.")
        print("\nNote: You may see compatibility warnings, but if all tests pass,")
        print("it means PyTorch can still use your GPU for training.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        print("You may need to install PyTorch nightly or build from source.")
    
    print()


if __name__ == "__main__":
    main()


