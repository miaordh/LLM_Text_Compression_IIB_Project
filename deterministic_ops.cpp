#include <torch/extension.h>

#include <cmath>
#include <vector>

namespace {

double kahan_sum(const double* data, int64_t n) {
    double sum = 0.0;
    double c = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        double y = data[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

} // namespace

torch::Tensor deterministic_softmax_forward(torch::Tensor logits) {
    auto cpu_logits = logits.to(torch::kCPU).to(torch::kFloat64).contiguous();

    const int64_t vocab = cpu_logits.size(-1);
    const int64_t rows = cpu_logits.numel() / vocab;

    auto probs = torch::empty_like(cpu_logits);
    const double* in_ptr = cpu_logits.data_ptr<double>();
    double* out_ptr = probs.data_ptr<double>();

    for (int64_t row = 0; row < rows; ++row) {
        const double* row_in = in_ptr + row * vocab;
        double* row_out = out_ptr + row * vocab;

        double max_val = row_in[0];
        for (int64_t j = 1; j < vocab; ++j) {
            if (row_in[j] > max_val) {
                max_val = row_in[j];
            }
        }

        std::vector<double> exps(vocab);
        for (int64_t j = 0; j < vocab; ++j) {
            exps[j] = std::exp(row_in[j] - max_val);
        }

        double sum_exp = kahan_sum(exps.data(), vocab);
        if (sum_exp == 0.0) {
            const double p = 1.0 / static_cast<double>(vocab);
            for (int64_t j = 0; j < vocab; ++j) {
                row_out[j] = p;
            }
            continue;
        }

        for (int64_t j = 0; j < vocab; ++j) {
            row_out[j] = exps[j] / sum_exp;
        }
    }

    return probs;
}

torch::Tensor deterministic_matmul_forward(torch::Tensor a, torch::Tensor b) {
    auto a_cpu = a.to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto b_cpu = b.to(torch::kCPU).to(torch::kFloat64).contiguous();

    TORCH_CHECK(a_cpu.dim() == 2, "deterministic_matmul: a must be 2D");
    TORCH_CHECK(b_cpu.dim() == 2, "deterministic_matmul: b must be 2D");

    const int64_t m = a_cpu.size(0);
    const int64_t k = a_cpu.size(1);
    TORCH_CHECK(b_cpu.size(0) == k, "deterministic_matmul: shape mismatch");
    const int64_t n = b_cpu.size(1);

    auto out = torch::zeros({m, n}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));

    const double* a_ptr = a_cpu.data_ptr<double>();
    const double* b_ptr = b_cpu.data_ptr<double>();
    double* out_ptr = out.data_ptr<double>();

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double sum = 0.0;
            double c = 0.0;
            for (int64_t kk = 0; kk < k; ++kk) {
                const double prod = a_ptr[i * k + kk] * b_ptr[kk * n + j];
                const double y = prod - c;
                const double t = sum + y;
                c = (t - sum) - y;
                sum = t;
            }
            out_ptr[i * n + j] = sum;
        }
    }

    return out;
}

torch::Tensor deterministic_rmsnorm_forward(torch::Tensor x, torch::Tensor weight, double eps) {
    auto x_cpu = x.to(torch::kCPU).to(torch::kFloat64).contiguous();
    auto w_cpu = weight.to(torch::kCPU).to(torch::kFloat64).contiguous();

    TORCH_CHECK(w_cpu.dim() == 1, "deterministic_rmsnorm: weight must be 1D");

    const int64_t hidden = w_cpu.size(0);
    TORCH_CHECK(x_cpu.size(-1) == hidden, "deterministic_rmsnorm: last dim mismatch");

    const int64_t rows = x_cpu.numel() / hidden;
    auto x_2d = x_cpu.view({rows, hidden});
    auto out_2d = torch::empty_like(x_2d);

    const double* x_ptr = x_2d.data_ptr<double>();
    const double* w_ptr = w_cpu.data_ptr<double>();
    double* out_ptr = out_2d.data_ptr<double>();

    for (int64_t row = 0; row < rows; ++row) {
        const double* r = x_ptr + row * hidden;
        double sq_sum = 0.0;
        double c = 0.0;
        for (int64_t j = 0; j < hidden; ++j) {
            const double sq = r[j] * r[j];
            const double y = sq - c;
            const double t = sq_sum + y;
            c = (t - sq_sum) - y;
            sq_sum = t;
        }

        const double mean_sq = sq_sum / static_cast<double>(hidden);
        const double inv_rms = 1.0 / std::sqrt(mean_sq + eps);

        double* out_row = out_ptr + row * hidden;
        for (int64_t j = 0; j < hidden; ++j) {
            out_row[j] = r[j] * inv_rms * w_ptr[j];
        }
    }

    return out_2d.view_as(x_cpu);
}

PYBIND11_MODULE(deterministic_ops_cpp, m) {
    m.def("softmax", &deterministic_softmax_forward, "Deterministic softmax (CPU float64)");
    m.def("matmul", &deterministic_matmul_forward, "Deterministic matmul (CPU float64)");
    m.def("rmsnorm", &deterministic_rmsnorm_forward, "Deterministic RMSNorm (CPU float64)");
}
