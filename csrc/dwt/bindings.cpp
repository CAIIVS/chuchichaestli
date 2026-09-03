// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <torch/extension.h>

// The GPU kernels are guarded by our own macro rather than torch's `USE_CUDA`
// or `USE_ROCM`, which a GPU-enabled torch defines for every translation unit
// it compiles, including a CPU-only build of this extension.

namespace c3li {

torch::Tensor dwt_axis_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                           const torch::Tensor& dec_hi, int64_t axis,
                           int64_t mode, int64_t pad_lo, int64_t out_length);

torch::Tensor idwt_axis_cpu(const torch::Tensor& coeffs,
                            const torch::Tensor& rec_lo,
                            const torch::Tensor& rec_hi, int64_t axis,
                            int64_t mode, int64_t trim, int64_t out_length);

#ifdef C3LI_WITH_GPU
torch::Tensor dwt_axis_cuda(const torch::Tensor& x, const torch::Tensor& dec_lo,
                            const torch::Tensor& dec_hi, int64_t axis,
                            int64_t mode, int64_t pad_lo, int64_t out_length);

torch::Tensor idwt_axis_cuda(const torch::Tensor& coeffs,
                             const torch::Tensor& rec_lo,
                             const torch::Tensor& rec_hi, int64_t axis,
                             int64_t mode, int64_t trim, int64_t out_length);
#endif

// Split every band along one spatial axis, on whichever device the input is on.
torch::Tensor dwt_axis(const torch::Tensor& x, const torch::Tensor& dec_lo,
                       const torch::Tensor& dec_hi, int64_t axis, int64_t mode,
                       int64_t pad_lo, int64_t out_length) {
#ifdef C3LI_WITH_GPU
  if (x.is_cuda()) {
    return dwt_axis_cuda(x, dec_lo, dec_hi, axis, mode, pad_lo, out_length);
  }
#endif
  return dwt_axis_cpu(x, dec_lo, dec_hi, axis, mode, pad_lo, out_length);
}

// Merge band pairs along one spatial axis, on whichever device the input is on.
torch::Tensor idwt_axis(const torch::Tensor& coeffs, const torch::Tensor& rec_lo,
                        const torch::Tensor& rec_hi, int64_t axis, int64_t mode,
                        int64_t trim, int64_t out_length) {
#ifdef C3LI_WITH_GPU
  if (coeffs.is_cuda()) {
    return idwt_axis_cuda(coeffs, rec_lo, rec_hi, axis, mode, trim, out_length);
  }
#endif
  return idwt_axis_cpu(coeffs, rec_lo, rec_hi, axis, mode, trim, out_length);
}

// Whether the extension carries kernels for the accelerator.
bool has_gpu() {
#ifdef C3LI_WITH_GPU
  return true;
#else
  return false;
#endif
}

}  // namespace c3li

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Discrete wavelet transforms with CPU and GPU kernels";
  m.def("dwt_axis", &c3li::dwt_axis, "Analysis along one spatial axis");
  m.def("idwt_axis", &c3li::idwt_axis, "Synthesis along one spatial axis");
  m.def("has_gpu", &c3li::has_gpu, "Whether GPU kernels were compiled in");
}
