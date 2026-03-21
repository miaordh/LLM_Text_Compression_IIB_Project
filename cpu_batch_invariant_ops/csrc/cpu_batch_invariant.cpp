#include <torch/extension.h>
#include <vector>
#include <cmath>

// Helper for deterministic block-chunked summation
template <typename scalar_t>
void deterministic_sum_cpu_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t batch_size,
    int64_t dim_size,
    int64_t block_size) 
{
    // Process each batch element sequentially or parallelly (but within the element, sequentially block-by-block)
    for (int64_t b = 0; b < batch_size; b++) {
        const scalar_t* row_input = input + b * dim_size;
        
        scalar_t total_sum = 0;
        int64_t num_blocks = (dim_size + block_size - 1) / block_size;
        
        for (int64_t block = 0; block < num_blocks; block++) {
            int64_t start_idx = block * block_size;
            int64_t end_idx = std::min(start_idx + block_size, dim_size);
            
            scalar_t block_sum = 0;
            // Strict sequential sum inside the block
            for (int64_t i = start_idx; i < end_idx; i++) {
                block_sum += row_input[i];
            }
            // Strict sequential sum of blocks
            total_sum += block_sum;
        }
        output[b] = total_sum;
    }
}

torch::Tensor mean_batch_invariant_cpu(torch::Tensor input, int64_t dim, bool keepdim = false, int64_t block_size = 128) {
    // Flatten dimensions before `dim` into a single batch dimension, and dimensions from `dim` to end into a single dim_size.
    // For simplicity, assuming dim is the last dimension (dim = -1) or similar typical usage for LayerNorm/RMSNorm
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim == input.dim() - 1, "Currently only supporting last dimension for batch invariant mean.");
    
    auto input_contig = input.contiguous();
    int64_t dim_size = input_contig.size(-1);
    int64_t batch_size = input_contig.numel() / dim_size;
    
    auto output_options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
    torch::Tensor output = torch::empty({batch_size}, output_options);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "mean_batch_invariant_cpu", ([&] {
        deterministic_sum_cpu_kernel<scalar_t>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            block_size
        );
    }));
    
    // Divide by dim_size to get mean
    output = output / dim_size;
    
    // Reshape output
    std::vector<int64_t> out_shape;
    for (int i = 0; i < input.dim() - 1; i++) {
        out_shape.push_back(input.size(i));
    }
    if (keepdim) {
        out_shape.push_back(1);
    }
    
    return output.view(out_shape);
}


template <typename scalar_t>
void deterministic_log_softmax_cpu_kernel(
    const scalar_t* input,
    scalar_t* output,
    int64_t batch_size,
    int64_t dim_size,
    int64_t block_size) 
{
    // Process each batch element
    for (int64_t b = 0; b < batch_size; b++) {
        const scalar_t* row_input = input + b * dim_size;
        scalar_t* row_output = output + b * dim_size;
        
        // 1. Blocked Max
        scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
        int64_t num_blocks = (dim_size + block_size - 1) / block_size;
        
        for (int64_t block = 0; block < num_blocks; block++) {
            int64_t start_idx = block * block_size;
            int64_t end_idx = std::min(start_idx + block_size, dim_size);
            scalar_t block_max = std::numeric_limits<scalar_t>::lowest();
            for (int64_t i = start_idx; i < end_idx; i++) {
                if (row_input[i] > block_max) block_max = row_input[i];
            }
            if (block_max > max_val) max_val = block_max;
        }

        // 2. Blocked sum of exp(x - max)
        scalar_t exp_sum = 0;
        for (int64_t block = 0; block < num_blocks; block++) {
            int64_t start_idx = block * block_size;
            int64_t end_idx = std::min(start_idx + block_size, dim_size);
            scalar_t block_sum = 0;
            for (int64_t i = start_idx; i < end_idx; i++) {
                block_sum += std::exp(row_input[i] - max_val);
            }
            exp_sum += block_sum;
        }
        
        scalar_t log_sum_exp = std::log(exp_sum) + max_val;
        
        // 3. Write output
        for (int64_t i = 0; i < dim_size; i++) {
            row_output[i] = row_input[i] - log_sum_exp;
        }
    }
}

torch::Tensor log_softmax_batch_invariant_cpu(torch::Tensor input, int64_t dim, int64_t block_size = 128) {
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim == input.dim() - 1, "Currently only supporting last dimension for batch invariant log_softmax.");
    
    auto input_contig = input.contiguous();
    int64_t dim_size = input_contig.size(-1);
    int64_t batch_size = input_contig.numel() / dim_size;
    
    torch::Tensor output = torch::empty_like(input_contig);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "log_softmax_batch_invariant_cpu", ([&] {
        deterministic_log_softmax_cpu_kernel<scalar_t>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim_size,
            block_size
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mean", &mean_batch_invariant_cpu, "Batch Invariant Mean (CPU)", py::arg("input"), py::arg("dim"), py::arg("keepdim") = false, py::arg("block_size") = 128);
    m.def("log_softmax", &log_softmax_batch_invariant_cpu, "Batch Invariant LogSoftmax (CPU)", py::arg("input"), py::arg("dim"), py::arg("block_size") = 128);
}
