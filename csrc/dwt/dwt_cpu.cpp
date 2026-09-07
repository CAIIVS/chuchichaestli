// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include <cstring>
#include <vector>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

namespace {

// Samples per tile of the contiguous axis; small enough for the accumulators to
// stay in registers, large enough to fill a vector unit.
constexpr int64_t kTile = 64;

}  // namespace

// One lane of the decomposition: `length` samples with `inner` between them,
// split into a low- and a high-pass half of `out_length`. `ext` is scratch the
// contiguous case writes its boundary extension into.
template <typename scalar_t, bool with_detail = true>
void dwt_plane(const scalar_t* lane, scalar_t* out_low, scalar_t* out_high,
               const scalar_t* lo, const scalar_t* hi, int64_t filter_len,
               int64_t length, int64_t out_length, int64_t inner, int64_t mode,
               int64_t pad_lo, scalar_t* ext, int64_t outer = 1) {
  using acc = acc_t<scalar_t>;
  using Vec = at::vec::Vectorized<scalar_t>;
  const int64_t width = Vec::size();
  const int64_t offset = filter_len - 1 - pad_lo;
  const int64_t extent = pad_lo + 2 * out_length + offset;
  (void)extent;
  const int64_t in_slab = length * inner;
  const int64_t out_slab = out_length * inner;
  for (int64_t ob = 0; ob < outer; ++ob, lane += in_slab,
               out_low += out_slab, out_high += out_slab) {

    // outputs whose taps all land inside the signal form one contiguous
    // run; splitting it out leaves the hot loop branch-free
    const int64_t interior_begin =
        std::max<int64_t>(0, (filter_len - offset) / 2);
    const int64_t interior_end =
        std::min(out_length, (length - offset + 1) / 2);

    if (inner == 1) {
      // in range every mode resolves to the sample itself, so only the
      // two ends need a rule
      const int64_t body = std::min(extent, pad_lo + length);
      for (int64_t j = 0; j < std::min(pad_lo, extent); ++j) {
        const PadRef ref = pad_resolve(j - pad_lo, length, mode);
        ext[j] = static_cast<scalar_t>(ref.sign) * lane[ref.index] +
                 static_cast<scalar_t>(ref.lo) * lane[0] +
                 static_cast<scalar_t>(ref.hi) * lane[length - 1];
      }
      if (body > pad_lo) {
        std::memcpy(ext + pad_lo, lane,
                    (body - pad_lo) * sizeof(scalar_t));
      }
      for (int64_t j = std::max<int64_t>(body, 0); j < extent; ++j) {
        const PadRef ref = pad_resolve(j - pad_lo, length, mode);
        ext[j] = static_cast<scalar_t>(ref.sign) * lane[ref.index] +
                 static_cast<scalar_t>(ref.lo) * lane[0] +
                 static_cast<scalar_t>(ref.hi) * lane[length - 1];
      }
      if constexpr (vectorizable<scalar_t>) {
        for (int64_t k = 0; k < out_length; k += width) {
          const scalar_t* tap = ext + 2 * k + offset + pad_lo;
          const int64_t rest = std::min<int64_t>(width, out_length - k);
          Vec acc_low(scalar_t(0));
          Vec acc_high(scalar_t(0));
          for (int64_t f = 0; f < filter_len; ++f) {
            const auto pair = at::vec::deinterleave2(
                Vec::loadu(tap - f), Vec::loadu(tap - f + width));
            acc_low = acc_low + Vec(lo[f]) * pair.first;
            if constexpr (with_detail) {
              acc_high = acc_high + Vec(hi[f]) * pair.first;
            }
          }
          acc_low.store(out_low + k, rest);
          if constexpr (with_detail) {
            acc_high.store(out_high + k, rest);
          }
        }
      } else {
        using acc = acc_t<scalar_t>;
        for (int64_t k = 0; k < out_length; ++k) {
          const scalar_t* tap = ext + 2 * k + offset + pad_lo;
          acc acc_low = acc(0);
          acc acc_high = acc(0);
          for (int64_t f = 0; f < filter_len; ++f) {
            const acc value = static_cast<acc>(tap[-f]);
            acc_low += static_cast<acc>(lo[f]) * value;
            if constexpr (with_detail) {
              acc_high += static_cast<acc>(hi[f]) * value;
            }
          }
          out_low[k] = static_cast<scalar_t>(acc_low);
          if constexpr (with_detail) {
            out_high[k] = static_cast<scalar_t>(acc_high);
          }
        }
      }
      continue;
    }

    for (int64_t k = 0; k < out_length; ++k) {
      const int64_t last = 2 * k + offset;
      const int64_t first = last - (filter_len - 1);
      const bool interior = k >= interior_begin && k < interior_end;
      scalar_t* low_row = out_low + k * inner;
      scalar_t* high_row = out_high + k * inner;

      if (vectorizable<scalar_t> && inner == 1 && interior) {
        // consecutive outputs read every other sample, so two loads and
        // a de-interleave give a whole vector of them
        const int64_t rest = std::min<int64_t>(width, interior_end - k);
        const int64_t take = 2 * rest;
        Vec acc_low(scalar_t(0));
        Vec acc_high(scalar_t(0));
        for (int64_t f = 0; f < filter_len; ++f) {
          const scalar_t* p = lane + 2 * k + offset - f;
          const Vec first = Vec::loadu(p, std::min<int64_t>(width, take));
          const Vec second = take > width
                                 ? Vec::loadu(p + width, take - width)
                                 : Vec(scalar_t(0));
          const auto pair = at::vec::deinterleave2(first, second);
          acc_low = acc_low + Vec(lo[f]) * pair.first;
          if constexpr (with_detail) {
            acc_high = acc_high + Vec(hi[f]) * pair.first;
          }
        }
        acc_low.store(low_row, rest);
        if constexpr (with_detail) {
          acc_high.store(high_row, rest);
        }
        k += rest - 1;
      } else if (inner == 1) {
        // the tail, and every output whose taps reach past an end
        using acc = acc_t<scalar_t>;
        const scalar_t* base = lane + last;
        acc acc_low = acc(0);
        acc acc_high = acc(0);
        if (interior) {
          for (int64_t f = 0; f < filter_len; ++f) {
            const acc value = static_cast<acc>(base[-f]);
            acc_low += static_cast<acc>(lo[f]) * value;
            if constexpr (with_detail) {
              acc_high += static_cast<acc>(hi[f]) * value;
            }
          }
        } else {
          for (int64_t f = 0; f < filter_len; ++f) {
            const PadRef ref = pad_resolve(last - f, length, mode);
            const acc value =
                static_cast<acc>(ref.sign) * static_cast<acc>(lane[ref.index]) +
                static_cast<acc>(ref.lo) * static_cast<acc>(lane[0]) +
                static_cast<acc>(ref.hi) * static_cast<acc>(lane[length - 1]);
            acc_low += static_cast<acc>(lo[f]) * value;
            if constexpr (with_detail) {
              acc_high += static_cast<acc>(hi[f]) * value;
            }
          }
        }
        *low_row = static_cast<scalar_t>(acc_low);
        if constexpr (with_detail) {
          *high_row = static_cast<scalar_t>(acc_high);
        }
      } else if (interior) {
        // every tap reads a sample that exists, so no rule is consulted
        using acc = acc_t<scalar_t>;
        const scalar_t* base = lane + last * inner;
        int64_t q = 0;
        if constexpr (vectorizable<scalar_t>) {
          for (; q + width <= inner; q += width) {
            Vec acc_low(scalar_t(0));
            Vec acc_high(scalar_t(0));
            for (int64_t f = 0; f < filter_len; ++f) {
              const Vec value = Vec::loadu(base - f * inner + q);
              acc_low = acc_low + Vec(lo[f]) * value;
              if constexpr (with_detail) {
                acc_high = acc_high + Vec(hi[f]) * value;
              }
            }
            acc_low.store(low_row + q);
            if constexpr (with_detail) {
              acc_high.store(high_row + q);
            }
          }
        }
        for (; q < inner; ++q) {
          acc acc_low = acc(0);
          acc acc_high = acc(0);
          for (int64_t f = 0; f < filter_len; ++f) {
            const acc value = static_cast<acc>(base[-f * inner + q]);
            acc_low += static_cast<acc>(lo[f]) * value;
            if constexpr (with_detail) {
              acc_high += static_cast<acc>(hi[f]) * value;
            }
          }
          low_row[q] = static_cast<scalar_t>(acc_low);
          if constexpr (with_detail) {
            high_row[q] = static_cast<scalar_t>(acc_high);
          }
        }
      } else {
        // the rules depend on the sample index alone, so each tap
        // resolves once and is reused across the contiguous axis
        const scalar_t* edge_lo = lane;
        const scalar_t* edge_hi = lane + (length - 1) * inner;
        using acc = acc_t<scalar_t>;
        for (int64_t q0 = 0; q0 < inner; q0 += kTile) {
          const int64_t span = std::min(kTile, inner - q0);
          acc acc_low[kTile] = {};
          acc acc_high[kTile] = {};
          for (int64_t f = 0; f < filter_len; ++f) {
            const PadRef ref = pad_resolve(last - f, length, mode);
            const acc sign = static_cast<acc>(ref.sign);
            const acc weight_lo = static_cast<acc>(ref.lo);
            const acc weight_hi = static_cast<acc>(ref.hi);
            const acc wl = static_cast<acc>(lo[f]);
            const acc wh = static_cast<acc>(hi[f]);
            // a zero-extended tap contributes nothing
            if (sign == acc(0) && weight_lo == acc(0) &&
                weight_hi == acc(0)) {
              continue;
            }
            const scalar_t* row = lane + ref.index * inner + q0;
            for (int64_t j = 0; j < span; ++j) {
              const acc value = sign * static_cast<acc>(row[j]) +
                                weight_lo * static_cast<acc>(edge_lo[q0 + j]) +
                                weight_hi * static_cast<acc>(edge_hi[q0 + j]);
              acc_low[j] += wl * value;
              if constexpr (with_detail) {
                acc_high[j] += wh * value;
              }
            }
          }
          for (int64_t j = 0; j < span; ++j) {
            low_row[q0 + j] = static_cast<scalar_t>(acc_low[j]);
            if constexpr (with_detail) {
              high_row[q0 + j] = static_cast<scalar_t>(acc_high[j]);
            }
          }
        }
      }
    }
  }
}

// Split every band of `x` along one spatial axis, without materializing the
// boundary extension. `(batch, groups, ...)` in, `(batch, 2 * groups, ...)`
// out, low-pass of group `g` at channel `2 g` and high-pass at `2 g + 1`.
//
// `out[k] = sum_f filter[f] * x_ext[2 k + offset - f]`, with
// `offset = filter_len - 1 - pad_lo`, matching the reference implementation.
// The nest keeps index arithmetic out of the innermost loop, which walks the
// contiguous axis and vectorizes.
torch::Tensor dwt_axis_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                           const torch::Tensor& dec_hi, int64_t axis,
                           int64_t mode, int64_t pad_lo, int64_t out_length,
                           c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  TORCH_CHECK(dec_lo.numel() == dec_hi.numel(),
              "the two decomposition filters must have the same length");

  const AxisLayout layout = axis_layout(x, axis);
  const int64_t filter_len = dec_lo.numel();
  const int64_t offset = filter_len - 1 - pad_lo;
  const int64_t length = layout.length;
  const int64_t inner = layout.inner;

  auto sizes = x.sizes().vec();
  sizes[1] *= 2;
  sizes[2 + axis] = out_length;
  // A caller that transforms one axis after another hands the same storage
  // back every time; allocating here instead would return freshly mapped
  // pages on every call, and faulting them in costs more than the transform.
  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value();
    TORCH_CHECK(out.sizes().vec() == sizes,
                "the output tensor does not have the shape the transform writes");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(),
                "the output tensor does not have the type of the input");
    C3LI_CHECK_CONTIGUOUS(out);
  } else {
    out = torch::empty(sizes, x.options());
  }

  const int64_t groups = x.size(1);
  const int64_t per_group = layout.outer / (x.size(0) * groups);

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "dwt_axis_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = dec_lo.to(x.scalar_type()).contiguous();
    const auto hi_filter = dec_hi.to(x.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();
    using Vec = at::vec::Vectorized<scalar_t>;
    const int64_t width = Vec::size();

    // Writing the extension out per lane leaves every output interior, so one
    // loop serves the axis; the slack keeps every load a whole vector.
    const int64_t extent = pad_lo + 2 * out_length + offset;

    // one task per lane: every index but the transformed axis
    parallel_for(layout.outer, [&](int64_t begin, int64_t end) {
      std::vector<scalar_t> ext(
          inner == 1 ? std::max<int64_t>(extent, 0) + 2 * width : 0);
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);

        const scalar_t* lane = src + o * length * inner;
        scalar_t* out_low =
            dst + ((batch * (2 * groups) + 2 * group) * per_group + pre) *
                      out_length * inner;
        scalar_t* out_high = out_low + per_group * out_length * inner;
        dwt_plane(lane, out_low, out_high, lo, hi, filter_len, length,
                  out_length, inner, mode, pad_lo, ext.data());
      }
    });
  });
  return out;
}

// Keep only the low-pass half of every band along one spatial axis.
// `(batch, groups, ...)` in, `(batch, groups, ...)` out.
//
// An approximation pyramid discards every detail band it is handed, and the
// channels it carries double on every axis that computes them. Skipping the
// high-pass leaves the arithmetic and the channel count where they started.
torch::Tensor dwt_lowpass_axis_cpu(const torch::Tensor& x,
                                   const torch::Tensor& dec_lo, int64_t axis,
                                   int64_t mode, int64_t pad_lo,
                                   int64_t out_length,
                                   c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);

  const AxisLayout layout = axis_layout(x, axis);
  const int64_t filter_len = dec_lo.numel();
  const int64_t offset = filter_len - 1 - pad_lo;
  const int64_t length = layout.length;
  const int64_t inner = layout.inner;

  auto sizes = x.sizes().vec();
  sizes[2 + axis] = out_length;
  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value();
    TORCH_CHECK(out.sizes().vec() == sizes,
                "the output tensor does not have the shape the transform writes");
    TORCH_CHECK(out.scalar_type() == x.scalar_type(),
                "the output tensor does not have the type of the input");
    C3LI_CHECK_CONTIGUOUS(out);
  } else {
    out = torch::empty(sizes, x.options());
  }

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "dwt_lowpass_axis_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = dec_lo.to(x.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    using Vec = at::vec::Vectorized<scalar_t>;
    const int64_t width = Vec::size();
    const int64_t extent = pad_lo + 2 * out_length + offset;

    parallel_for(layout.outer, [&](int64_t begin, int64_t end) {
      std::vector<scalar_t> ext(
          inner == 1 ? std::max<int64_t>(extent, 0) + 2 * width : 0);
      for (int64_t o = begin; o < end; ++o) {
        const scalar_t* lane = src + o * length * inner;
        scalar_t* out_low = dst + o * out_length * inner;
        dwt_plane<scalar_t, false>(lane, out_low, out_low, lo, lo, filter_len,
                                   length, out_length, inner, mode, pad_lo,
                                   ext.data());
      }
    });
  });
  return out;
}

// Split every band over every spatial axis in one pass.
// `(batch, groups, ...)` in, `(batch, 2**d * groups, ...)` out.
//
// The per-axis form writes a whole tensor between axes, so a two-dimensional
// decomposition streams its result through memory three times over. Here one
// lane is carried through every axis in a scratch buffer that stays in cache,
// and only the finished lane is written out.
torch::Tensor dwt_nd_cpu(const torch::Tensor& x, const torch::Tensor& dec_lo,
                         const torch::Tensor& dec_hi, int64_t mode,
                         const std::vector<int64_t>& pad_los,
                         const std::vector<int64_t>& out_lengths) {
  C3LI_CHECK_CONTIGUOUS(x);
  C3LI_CHECK_FLOATING(x);
  TORCH_CHECK(dec_lo.numel() == dec_hi.numel(),
              "the two decomposition filters must have the same length");
  const int64_t dimensions = static_cast<int64_t>(out_lengths.size());
  TORCH_CHECK(dimensions >= 1 && dimensions <= 3,
              "a fused decomposition runs over one to three axes");
  TORCH_CHECK(static_cast<int64_t>(pad_los.size()) == dimensions,
              "every axis needs a padding");
  TORCH_CHECK(x.dim() == dimensions + 2,
              "the input must carry a batch and a channel axis");

  const int64_t corners = int64_t{1} << dimensions;
  const int64_t groups = x.size(1);
  const int64_t batch = x.size(0);
  const int64_t filter_len = dec_lo.numel();

  std::vector<int64_t> in_shape(dimensions);
  for (int64_t d = 0; d < dimensions; ++d) {
    in_shape[d] = x.size(2 + d);
  }

  auto sizes = x.sizes().vec();
  sizes[1] = corners * groups;
  for (int64_t d = 0; d < dimensions; ++d) {
    sizes[2 + d] = out_lengths[d];
  }
  torch::Tensor out = torch::empty(sizes, x.options());

  // the widest a stage gets: axes already done are output length, the rest
  // are still input length
  int64_t scratch = 0;
  for (int64_t stage = 0; stage <= dimensions; ++stage) {
    int64_t extent = int64_t{1} << stage;
    for (int64_t d = 0; d < dimensions; ++d) {
      extent *= (d < stage) ? out_lengths[d] : in_shape[d];
    }
    scratch = std::max(scratch, extent);
  }

  C3LI_DISPATCH_FLOATING(x.scalar_type(), "dwt_nd_cpu", [&] {
    const auto* src = x.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = dec_lo.to(x.scalar_type()).contiguous();
    const auto hi_filter = dec_hi.to(x.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();

    int64_t in_lane = 1;
    for (int64_t d = 0; d < dimensions; ++d) {
      in_lane *= in_shape[d];
    }
    int64_t out_lane = 1;
    for (int64_t d = 0; d < dimensions; ++d) {
      out_lane *= out_lengths[d];
    }
    int64_t widest_ext = 0;
    for (int64_t d = 0; d < dimensions; ++d) {
      const int64_t offset = filter_len - 1 - pad_los[d];
      widest_ext = std::max(widest_ext, pad_los[d] + 2 * out_lengths[d] + offset);
    }

    parallel_for(batch * groups, [&](int64_t begin, int64_t end) {
      // kept across calls: a lane's working set is small and reallocating it
      // per chunk costs more than the split does on a short axis
      static thread_local std::vector<scalar_t> front_buf;
      static thread_local std::vector<scalar_t> back_buf;
      static thread_local std::vector<scalar_t> ext_buf;
      const int64_t ext_want =
          widest_ext + 2 * at::vec::Vectorized<scalar_t>::size();
      if (static_cast<int64_t>(front_buf.size()) < scratch) {
        front_buf.resize(scratch);
        back_buf.resize(scratch);
      }
      if (static_cast<int64_t>(ext_buf.size()) < ext_want) {
        ext_buf.resize(ext_want);
      }
      std::vector<scalar_t>& front = front_buf;
      std::vector<scalar_t>& back = back_buf;

      for (int64_t o = begin; o < end; ++o) {
        const int64_t group = o % groups;
        const int64_t b = o / groups;
        const scalar_t* lane = src + (b * groups + group) * in_lane;
        std::copy(lane, lane + in_lane, front.data());

        // axes split from the first outwards, doubling the planes each time.
        // A plane still carries the axes not yet done, so the ones outside
        // this axis are walked here.
        int64_t plane_len = in_lane;
        int64_t planes = 1;
        for (int64_t axis = 0; axis < dimensions; ++axis) {
          const int64_t length = in_shape[axis];
          const int64_t wanted = out_lengths[axis];
          int64_t inner = 1;
          for (int64_t d = axis + 1; d < dimensions; ++d) {
            inner *= in_shape[d];
          }
          const int64_t outer = plane_len / (length * inner);
          const int64_t split = outer * wanted * inner;
          for (int64_t pl = 0; pl < planes; ++pl) {
            const scalar_t* from = front.data() + pl * plane_len;
            scalar_t* low = back.data() + (2 * pl) * split;
            scalar_t* high = back.data() + (2 * pl + 1) * split;
            dwt_plane(from, low, high, lo, hi, filter_len, length, wanted,
                      inner, mode, pad_los[axis], ext_buf.data(), outer);
          }
          front.swap(back);
          plane_len = split;
          planes *= 2;
        }

        // the split interleaves low and high per plane, which is the order
        // the subband names are read in
        for (int64_t c = 0; c < corners; ++c) {
          std::copy(front.data() + c * out_lane,
                    front.data() + (c + 1) * out_lane,
                    dst + ((b * groups + group) * corners + c) * out_lane);
        }
      }
    });
  });
  return out;
}

}  // namespace c3li
