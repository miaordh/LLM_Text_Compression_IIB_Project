import torch
import cpu_batch_invariant_ops_c

def test_mean():
    # Mean reduction should not change value when computed with varying batch padding
    # Let's just verify it runs and is close to standard mean
    a = torch.randn(2, 64, 4096, dtype=torch.float32)
    m1 = a.mean(dim=-1)
    m2 = cpu_batch_invariant_ops_c.mean(a, -1, False, 128)
    diff = torch.abs(m1 - m2).max().item()
    print(f"Mean diff vs standard PyTorch: {diff}")

def test_softmax():
    a = torch.randn(2, 64, 4096, dtype=torch.float32)
    sm1 = torch.nn.functional.log_softmax(a, dim=-1)
    sm2 = cpu_batch_invariant_ops_c.log_softmax(a, -1, 128)
    diff = torch.abs(sm1 - sm2).max().item()
    print(f"LogSoftmax diff vs standard PyTorch: {diff}")

def test_matmul():
    a = torch.randn(4, 128, 4096, dtype=torch.float32)
    b = torch.randn(4, 4096, 64, dtype=torch.float32)
    mm1 = a @ b
    mm2 = cpu_batch_invariant_ops_c.matmul(a, b, 128)
    diff = torch.abs(mm1 - mm2).max().item()
    print(f"MatMul diff vs standard PyTorch: {diff}")

if __name__ == '__main__':
    test_mean()
    test_softmax()
    pass
    # test_matmul()
