import torch
import cpu_batch_invariant_ops_c

def _cpu_batch_invariant_mean(input, dim, keepdim=False):
    """
    Python wrapper for the C++ CPU block-chunked deterministic mean.
    """
    return cpu_batch_invariant_ops_c.mean(input, dim, keepdim)

def _cpu_batch_invariant_log_softmax(input, dim=-1):
    """
    Python wrapper for the C++ CPU block-chunked deterministic log_softmax.
    """
    return cpu_batch_invariant_ops_c.log_softmax(input, dim)

def patch_llama_for_cpu_determinism():
    """
    Monkey-patches HuggingFace LLaMA internals to use our deterministic CPU ops.
    """
    import transformers.models.llama.modeling_llama as modeling_llama

    # 1. Patch RMSNorm
    original_rmsnorm_forward = modeling_llama.LlamaRMSNorm.forward

    def deterministic_rmsnorm_forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # Use our deterministic mean instead of variance = hidden_states.pow(2).mean(-1, keepdim=True)
        variance = _cpu_batch_invariant_mean(hidden_states.pow(2), dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    modeling_llama.LlamaRMSNorm.forward = deterministic_rmsnorm_forward

    # 2. Patch Softmax in Attention
    # We'll patch the standard attention forward where it does F.softmax
    # Note: Transformers usually uses nn.functional.softmax, we can override or patch the module level.
    if hasattr(modeling_llama, "F"):
        original_softmax = modeling_llama.F.softmax
        
        def deterministic_softmax(input, dim=-1, dtype=None):
            # Softmax = exp(log_softmax), using our deterministic C++ log_softmax
            if input.device.type == 'cpu':
                log_sm = _cpu_batch_invariant_log_softmax(input.float(), dim=dim)
                res = torch.exp(log_sm)
                if dtype is not None:
                    res = res.to(dtype)
                elif input.dtype != torch.float32:
                    res = res.to(input.dtype)
                return res
            else:
                return original_softmax(input, dim=dim, dtype=dtype)
        
        # Override the F.softmax alias inside modeling_llama
        modeling_llama.F.softmax = deterministic_softmax

    print("Successfully patched Hugging Face LLaMA for CPU determinism using C++ block invariant ops.")

if __name__ == "__main__":
    # Quick test
    x = torch.randn(2, 4096)
    m1 = x.mean(dim=-1)
    m2 = _cpu_batch_invariant_mean(x, dim=-1)
    print("Mean diff:", torch.abs(m1 - m2).max().item())

    sm1 = torch.nn.functional.log_softmax(x, dim=-1)
    sm2 = _cpu_batch_invariant_log_softmax(x, dim=-1)
    print("LogSoftmax diff:", torch.abs(sm1 - sm2).max().item())
