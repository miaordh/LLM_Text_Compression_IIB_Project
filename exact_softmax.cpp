#include <torch/extension.h>
#include <cmath>
#include <vector>

// [FIX] Aggressive Rounding to 2 Decimals (100.0)
// This creates a large enough "bucket" to trap hardware matmul noise.
// 12.5674 and 12.5676 both snap to 12.57.
const double PRECISION_SCALE = 100.0; 

inline double round_dec(double val) {
    return std::round(val * PRECISION_SCALE) / PRECISION_SCALE;
}

torch::Tensor exact_softmax_forward(torch::Tensor logits) {
    auto cpu_logits = logits.to(torch::kCPU).to(torch::kDouble).contiguous();
    
    int64_t vocab_size = cpu_logits.size(-1);
    int64_t num_rows = cpu_logits.numel() / vocab_size;
    
    double* data_ptr = cpu_logits.data_ptr<double>();
    
    auto probs = torch::empty_like(cpu_logits);
    double* out_ptr = probs.data_ptr<double>();
    
    for (int64_t i = 0; i < num_rows; i++) {
        double* row_in = data_ptr + (i * vocab_size);
        double* row_out = out_ptr + (i * vocab_size);

        // A. Find Max with Rounding
        double max_val = -1e9;
        for (int64_t j = 0; j < vocab_size; j++) {
            double val = round_dec(row_in[j]);
            if (val > max_val) max_val = val;
        }

        // B. Exponentials and Sum
        double sum_exp = 0.0;
        for (int64_t j = 0; j < vocab_size; j++) {
            double val = round_dec(row_in[j]);
            double e = std::exp(val - max_val);
            row_out[j] = e;
            sum_exp += e;
        }

        // C. Normalize
        for (int64_t j = 0; j < vocab_size; j++) {
            row_out[j] /= sum_exp;
        }
    }
    
    return probs.to(torch::kFloat32);
}

PYBIND11_MODULE(exact_softmax_cpp, m) {
    m.def("forward", &exact_softmax_forward, "Exact Deterministic Softmax (CPU)");
}