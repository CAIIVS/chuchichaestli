// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/record_function.h>
#include <torch/extension.h>

#include <cstring>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

// The GPU kernels are guarded by our own macro, not torch's `USE_CUDA` or
// `USE_ROCM`: a GPU-enabled torch defines those even for a CPU-only build of
// this extension.
//
// `RECORD_FUNCTION` is the marker ATen puts on its own operators, so
// `torch.profiler` attributes the time here rather than to an opaque Python
// frame. It reads one flag and returns when nothing is profiling.

namespace c3li {

torch::Tensor dwt_lift_axis_cpu(
    const torch::Tensor& x, int64_t axis,
    const std::vector<int64_t>& on_detail,
    const std::vector<std::vector<double>>& coeffs,
    const std::vector<int64_t>& lows, double approx_gain, int64_t approx_delay,
    double detail_gain, int64_t detail_delay,
    c10::optional<torch::Tensor> out_opt);

torch::Tensor dwt_axis_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                           const torch::Tensor& dec_hi, int64_t axis,
                           int64_t mode, int64_t pad_lo, int64_t out_length,
                           c10::optional<torch::Tensor> out_opt);

torch::Tensor dwt_lowpass_axis_cpu(const torch::Tensor& x,
                                   const torch::Tensor& dec_lo, int64_t axis,
                                   int64_t mode, int64_t pad_lo,
                                   int64_t out_length,
                                   c10::optional<torch::Tensor> out_opt);

torch::Tensor dwt_nd_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                         const torch::Tensor& dec_hi, int64_t mode,
                         const std::vector<int64_t>& pad_los,
                         const std::vector<int64_t>& out_lengths,
                         c10::optional<torch::Tensor> out_opt);

torch::Tensor idwt_axis_cpu(const torch::Tensor& coeffs,
                            const torch::Tensor& rec_lo,
                            const torch::Tensor& rec_hi, int64_t axis,
                            int64_t mode, int64_t trim, int64_t out_length,
                            c10::optional<torch::Tensor> out_opt);

torch::Tensor idwt_nd_cpu(const torch::Tensor& coeffs,
                          const torch::Tensor& rec_lo,
                          const torch::Tensor& rec_hi, int64_t mode,
                          const std::vector<int64_t>& trims,
                          const std::vector<int64_t>& out_lengths,
                          c10::optional<torch::Tensor> out_opt);

torch::Tensor haar_nd_cpu(const torch::Tensor& x, double scale,
                          c10::optional<torch::Tensor> out_opt);

#ifdef C3LI_WITH_GPU
torch::Tensor dwt_axis_cuda(const torch::Tensor& x, const torch::Tensor& dec_lo,
                            const torch::Tensor& dec_hi, int64_t axis,
                            int64_t mode, int64_t pad_lo, int64_t out_length,
                            c10::optional<torch::Tensor> out_opt);

torch::Tensor idwt_axis_cuda(const torch::Tensor& coeffs,
                             const torch::Tensor& rec_lo,
                             const torch::Tensor& rec_hi, int64_t axis,
                             int64_t mode, int64_t trim, int64_t out_length,
                             c10::optional<torch::Tensor> out_opt);

torch::Tensor dwt_lowpass_axis_cuda(const torch::Tensor& x,
                                    const torch::Tensor& dec_lo, int64_t axis,
                                    int64_t mode, int64_t pad_lo,
                                    int64_t out_length,
                                    c10::optional<torch::Tensor> out_opt);

torch::Tensor haar_nd_cuda(const torch::Tensor& x, double scale,
                           c10::optional<torch::Tensor> out_opt);
#endif

// Split every band along one spatial axis, on whichever device the input is on.
torch::Tensor dwt_axis(const torch::Tensor& x, const torch::Tensor& dec_lo,
                       const torch::Tensor& dec_hi, int64_t axis, int64_t mode,
                       int64_t pad_lo, int64_t out_length,
                       c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::dwt_axis", std::vector<c10::IValue>());
#ifdef C3LI_WITH_GPU
  if (x.is_cuda()) {
    return dwt_axis_cuda(x, dec_lo, dec_hi, axis, mode, pad_lo, out_length,
                         out);
  }
#endif
  return dwt_axis_cpu(x, dec_lo, dec_hi, axis, mode, pad_lo, out_length, out);
}

// Keep the low-pass half along one spatial axis, on whichever device the
// input is on.
torch::Tensor dwt_lowpass_axis(const torch::Tensor& x,
                               const torch::Tensor& dec_lo, int64_t axis,
                               int64_t mode, int64_t pad_lo,
                               int64_t out_length,
                               c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::dwt_lowpass_axis", std::vector<c10::IValue>());
#ifdef C3LI_WITH_GPU
  if (x.is_cuda()) {
    return dwt_lowpass_axis_cuda(x, dec_lo, axis, mode, pad_lo, out_length,
                                 out);
  }
#endif
  return dwt_lowpass_axis_cpu(x, dec_lo, axis, mode, pad_lo, out_length, out);
}

// Merge band pairs along one spatial axis, on whichever device the input is on.
torch::Tensor idwt_axis(const torch::Tensor& coeffs, const torch::Tensor& rec_lo,
                        const torch::Tensor& rec_hi, int64_t axis, int64_t mode,
                        int64_t trim, int64_t out_length,
                        c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::idwt_axis", std::vector<c10::IValue>());
#ifdef C3LI_WITH_GPU
  if (coeffs.is_cuda()) {
    return idwt_axis_cuda(coeffs, rec_lo, rec_hi, axis, mode, trim, out_length,
                          out);
  }
#endif
  return idwt_axis_cpu(coeffs, rec_lo, rec_hi, axis, mode, trim, out_length, out);
}

// Transform every spatial axis at once with the Haar wavelet.
torch::Tensor haar_nd(const torch::Tensor& x, double scale,
                      c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::haar_nd", std::vector<c10::IValue>());
#ifdef C3LI_WITH_GPU
  if (x.is_cuda()) {
    return haar_nd_cuda(x, scale, out);
  }
#endif
  return haar_nd_cpu(x, scale, out);
}

// Split every band along every spatial axis, on whichever device the input is
// on. The host fuses the axes into one pass; the accelerator has a kernel per
// axis and takes them in turn.
torch::Tensor dwt_nd(const torch::Tensor& x, const torch::Tensor& dec_lo,
                     const torch::Tensor& dec_hi, int64_t mode,
                     const std::vector<int64_t>& pad_los,
                     const std::vector<int64_t>& out_lengths,
                     c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::dwt_nd", std::vector<c10::IValue>());
  const int64_t dimensions = x.dim() - 2;
  TORCH_CHECK(static_cast<int64_t>(pad_los.size()) == dimensions &&
                  static_cast<int64_t>(out_lengths.size()) == dimensions,
              "a padding and a length are needed for every spatial axis");
#ifdef C3LI_WITH_GPU
  if (x.is_cuda()) {
    // the axes run in turn, so the storage handed in takes the last result
    // and the ones before it are the kernel's own
    torch::Tensor bands = x;
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      const bool last = axis + 1 == dimensions;
      bands = dwt_axis_cuda(bands, dec_lo, dec_hi, axis, mode, pad_los[axis],
                            out_lengths[axis], last ? out : c10::nullopt);
    }
    return bands;
  }
#endif
  return dwt_nd_cpu(x, dec_lo, dec_hi, mode, pad_los, out_lengths, out);
}

// Merge every band pair along every spatial axis, on whichever device the
// input is on. The axes unwind in the order a decomposition laid them down,
// so the last one split is the first one merged.
torch::Tensor idwt_nd(const torch::Tensor& coeffs, const torch::Tensor& rec_lo,
                      const torch::Tensor& rec_hi, int64_t mode,
                      const std::vector<int64_t>& trims,
                      const std::vector<int64_t>& out_lengths,
                      c10::optional<torch::Tensor> out = c10::nullopt) {
  RECORD_FUNCTION("c3li::idwt_nd", std::vector<c10::IValue>());
  const int64_t dimensions = coeffs.dim() - 2;
  TORCH_CHECK(static_cast<int64_t>(trims.size()) == dimensions &&
                  static_cast<int64_t>(out_lengths.size()) == dimensions,
              "a trim and a length are needed for every spatial axis");
#ifdef C3LI_WITH_GPU
  if (coeffs.is_cuda()) {
    torch::Tensor bands = coeffs;
    for (int64_t axis = dimensions - 1; axis >= 0; --axis) {
      bands = idwt_axis_cuda(bands, rec_lo, rec_hi, axis, mode, trims[axis],
                             out_lengths[axis], axis == 0 ? out : c10::nullopt);
    }
    return bands;
  }
#endif
  return idwt_nd_cpu(coeffs, rec_lo, rec_hi, mode, trims, out_lengths,
                     out);
}

// Lift the approximation band of every group out of `bands`, whose channels
// run `g * corners + b`, so that it can feed the next level.
//
// On the host a strided `slice(...).contiguous()` turns whole-lane copies into
// an element-wise gather that costs more than the transform itself, so the
// copy is written out instead.
static torch::Tensor approximation(const torch::Tensor& bands, int64_t groups,
                                   int64_t corners) {
  if (!bands.is_cpu()) {
    return bands.slice(1, 0, groups * corners, corners).contiguous();
  }
  auto sizes = bands.sizes().vec();
  sizes[1] = groups;
  torch::Tensor out = torch::empty(sizes, bands.options());
  const int64_t batch = bands.size(0);
  const int64_t channels = bands.size(1);
  const int64_t lane = bands.numel() / (batch * channels);

  C3LI_DISPATCH_FLOATING(bands.scalar_type(), "approximation", [&] {
    const auto* src = bands.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    parallel_for(batch * groups, [&](int64_t begin, int64_t end) {
      for (int64_t i = begin; i < end; ++i) {
        const int64_t b = i / groups;
        const int64_t g = i % groups;
        std::memcpy(dst + i * lane,
                    src + (b * channels + g * corners) * lane,
                    lane * sizeof(scalar_t));
      }
    });
  });
  return out;
}

// Transform every axis repeatedly, each level working on the last
// approximation. The lengths and paddings a level needs follow from the one
// before it, so running the recursion here saves a round trip per level.
std::vector<torch::Tensor> wavedec_axes(const torch::Tensor& x,
                                        const torch::Tensor& dec_lo,
                                        const torch::Tensor& dec_hi, int64_t mode,
                                        int64_t levels) {
  RECORD_FUNCTION("c3li::wavedec_axes", std::vector<c10::IValue>());
  TORCH_CHECK(levels >= 1, "a decomposition needs at least one level");
  const int64_t dimensions = x.dim() - 2;
  const int64_t filter_len = dec_lo.numel();
  const int64_t corners = int64_t{1} << dimensions;
  const int64_t groups = x.size(1);

  std::vector<torch::Tensor> stacked;
  stacked.reserve(levels);
  torch::Tensor current = x;
  for (int64_t level = 0; level < levels; ++level) {
    torch::Tensor bands = current;
    for (int64_t axis = 0; axis < dimensions; ++axis) {
      const int64_t length = bands.size(2 + axis);
      int64_t pad_lo;
      int64_t out_length;
      if (mode == kPeriodization) {  // the one critically sampled mode
        TORCH_CHECK(length % 2 == 0,
                    "the fused recursion needs even axes for this mode");
        pad_lo = filter_len / 2 - 1;
        out_length = length / 2;
      } else {
        pad_lo = filter_len - 2;
        out_length = (length + filter_len - 1) / 2;
      }
      bands = dwt_axis(bands, dec_lo, dec_hi, axis, mode, pad_lo, out_length);
    }
    stacked.push_back(bands);
    if (level + 1 < levels) {
      current = approximation(bands, groups, corners);
    }
  }
  return stacked;
}

// Transform repeatedly, each level working on the approximation of the last.
std::vector<torch::Tensor> haar_wavedec(const torch::Tensor& x, int64_t levels,
                                        double scale) {
  RECORD_FUNCTION("c3li::haar_wavedec", std::vector<c10::IValue>());
  TORCH_CHECK(levels >= 1, "a decomposition needs at least one level");
  const int64_t corners = int64_t{1} << (x.dim() - 2);
  const int64_t groups = x.size(1);

  std::vector<torch::Tensor> stacked;
  stacked.reserve(levels);
  torch::Tensor current = x;
  for (int64_t level = 0; level < levels; ++level) {
    torch::Tensor bands = haar_nd(current, scale, c10::nullopt);
    stacked.push_back(bands);
    if (level + 1 < levels) {
      current = approximation(bands, groups, corners);
    }
  }
  return stacked;
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
  m.def("dwt_lift_axis", &c3li::dwt_lift_axis_cpu,
        "Decomposition along one spatial axis by lifting", py::arg("x"),
        py::arg("axis"), py::arg("on_detail"), py::arg("coeffs"),
        py::arg("lows"), py::arg("approx_gain"), py::arg("approx_delay"),
        py::arg("detail_gain"), py::arg("detail_delay"),
        py::arg("out") = py::none());
  m.def("dwt_axis", &c3li::dwt_axis, "Decomposition along one spatial axis",
        py::arg("x"), py::arg("dec_lo"), py::arg("dec_hi"), py::arg("axis"),
        py::arg("mode"), py::arg("pad_lo"), py::arg("out_length"),
        py::arg("out") = py::none());
  m.def("idwt_axis", &c3li::idwt_axis, "Reconstruction along one spatial axis",
        py::arg("coeffs"), py::arg("rec_lo"), py::arg("rec_hi"), py::arg("axis"),
        py::arg("mode"), py::arg("trim"), py::arg("out_length"),
        py::arg("out") = py::none());
  m.def("haar_nd", &c3li::haar_nd, "Fused Haar decomposition over every axis",
        py::arg("x"), py::arg("scale"), py::arg("out") = py::none());
  m.def("wavedec_axes", &c3li::wavedec_axes,
        "Fused multi-level decomposition over every axis");
  m.def("haar_wavedec", &c3li::haar_wavedec,
        "Fused multi-level Haar decomposition");
  m.def("dwt_lowpass_axis", &c3li::dwt_lowpass_axis,
        "Low-pass half of a decomposition along one spatial axis",
        py::arg("x"), py::arg("dec_lo"), py::arg("axis"), py::arg("mode"),
        py::arg("pad_lo"), py::arg("out_length"), py::arg("out") = py::none());
  m.def("dwt_nd", &c3li::dwt_nd, "Fused decomposition over every axis",
        py::arg("x"), py::arg("dec_lo"), py::arg("dec_hi"), py::arg("mode"),
        py::arg("pad_los"), py::arg("out_lengths"),
        py::arg("out") = py::none());
  m.def("idwt_nd", &c3li::idwt_nd, "Fused reconstruction over every axis",
        py::arg("coeffs"), py::arg("rec_lo"), py::arg("rec_hi"),
        py::arg("mode"), py::arg("trims"), py::arg("out_lengths"),
        py::arg("out") = py::none());
  m.def("has_gpu", &c3li::has_gpu, "Whether GPU kernels were compiled in");
  m.def("supported_dtypes", &c3li::supported_dtypes,
        "The tensor types the kernels were instantiated for");
}
