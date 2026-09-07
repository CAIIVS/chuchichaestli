// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#include <ATen/cpu/vec/vec.h>

#include <vector>

#include "../common/boundary.h"
#include "../common/dispatch.h"
#include "../common/parallel.h"

namespace c3li {

// A transposed convolution puts the contribution of coefficient `k` and tap
// `f` at `2 k + f`, so gathering the output at `t` means collecting every
// `(k, f)` with `2 k + f == t + trim`. That pairs `t` with one phase of the
// filter: `f` shares the parity of `t + trim`, and the coefficients it reads
// run backwards from `(t + trim - parity) / 2`. Splitting the taps by parity
// turns the gather into two ordinary correlations with nothing to test in the
// inner loop. The critically sampled mode wraps the sum around the output
// instead of discarding the overhang.
template <typename scalar_t>
void idwt_plane(const scalar_t* low, const scalar_t* high, scalar_t* out,
                const scalar_t* lo, const scalar_t* hi, int64_t filter_len,
                int64_t coeff_length, int64_t out_length, int64_t inner,
                int64_t trim, bool circular, int64_t outer = 1) {
  using acc = acc_t<scalar_t>;
  using Vec = at::vec::Vectorized<scalar_t>;
  const int64_t width = Vec::size();
  const int64_t taps[2] = {(filter_len + 1) / 2, filter_len / 2};

  const auto gather = [&](int64_t t, int64_t q) {
    const int64_t s = t + trim;
    const int64_t p = s & 1;
    const int64_t first = (s - p) / 2;
    acc value = acc(0);
    for (int64_t j = 0; j < taps[p]; ++j) {
      int64_t k = first - j;
      if (circular) {
        k = ((k % coeff_length) + coeff_length) % coeff_length;
      } else if (k < 0 || k >= coeff_length) {
        continue;
      }
      value += static_cast<acc>(lo[2 * j + p]) *
                   static_cast<acc>(low[k * inner + q]) +
               static_cast<acc>(hi[2 * j + p]) *
                   static_cast<acc>(high[k * inner + q]);
    }
    return value;
  };

  const int64_t widest = std::max(taps[0], taps[1]);
  int64_t interior_begin = std::max<int64_t>(2 * (widest - 1) - trim + 1, 0);
  int64_t interior_end = std::min(2 * coeff_length - trim - 1, out_length);
  if (interior_end < interior_begin) {
    interior_end = interior_begin;
  }

  const int64_t in_slab = coeff_length * inner;
  const int64_t out_slab = out_length * inner;
  for (int64_t ob = 0; ob < outer; ++ob, low += in_slab, high += in_slab,
               out += out_slab) {
  for (int64_t t = 0; t < interior_begin; ++t) {
    for (int64_t q = 0; q < inner; ++q) {
      out[t * inner + q] = static_cast<scalar_t>(gather(t, q));
    }
  }
  for (int64_t t = interior_end; t < out_length; ++t) {
    for (int64_t q = 0; q < inner; ++q) {
      out[t * inner + q] = static_cast<scalar_t>(gather(t, q));
    }
  }

  if (inner == 1) {
    int64_t t = interior_begin;
    if constexpr (vectorizable<scalar_t>) {
      // consecutive outputs of one parity read consecutive coefficients, so
      // each parity loads whole vectors and the two interleave. A short axis
      // is mostly tail, so the last block is masked rather than left to the
      // scalar loop.
      while (t < interior_end) {
        const int64_t rest = std::min<int64_t>(2 * width, interior_end - t);
        Vec part[2];
        for (int64_t r = 0; r < 2; ++r) {
          const int64_t count = (rest - r + 1) / 2;
          if (count <= 0) {
            part[r] = Vec(scalar_t(0));
            continue;
          }
          const int64_t s = t + r + trim;
          const int64_t p = s & 1;
          const int64_t first = (s - p) / 2;
          Vec sum_lo(scalar_t(0));
          Vec sum_hi(scalar_t(0));
          for (int64_t j = 0; j < taps[p]; ++j) {
            const int64_t k = first - j;
            sum_lo = at::vec::fmadd(Vec(lo[2 * j + p]),
                                    Vec::loadu(low + k, count), sum_lo);
            sum_hi = at::vec::fmadd(Vec(hi[2 * j + p]),
                                    Vec::loadu(high + k, count), sum_hi);
          }
          part[r] = sum_lo + sum_hi;
        }
        const auto woven = at::vec::interleave2(part[0], part[1]);
        woven.first.store(out + t, std::min<int64_t>(width, rest));
        if (rest > width) {
          woven.second.store(out + t + width, rest - width);
        }
        t += rest;
      }
    }
    for (; t < interior_end; ++t) {
      out[t] = static_cast<scalar_t>(gather(t, 0));
    }
    continue;
  }

  for (int64_t t = interior_begin; t < interior_end; ++t) {
    const int64_t s = t + trim;
    const int64_t p = s & 1;
    const int64_t first = (s - p) / 2;
    scalar_t* row = out + t * inner;
    int64_t q = 0;
    if constexpr (vectorizable<scalar_t>) {
      // the axis is strided, so a whole vector of the trailing one is read
      // for every tap
      for (; q + width <= inner; q += width) {
        Vec sum_lo(scalar_t(0));
        Vec sum_hi(scalar_t(0));
        for (int64_t j = 0; j < taps[p]; ++j) {
          const int64_t k = first - j;
          sum_lo = at::vec::fmadd(Vec(lo[2 * j + p]),
                                  Vec::loadu(low + k * inner + q), sum_lo);
          sum_hi = at::vec::fmadd(Vec(hi[2 * j + p]),
                                  Vec::loadu(high + k * inner + q), sum_hi);
        }
        (sum_lo + sum_hi).store(row + q);
      }
    }
    for (; q < inner; ++q) {
      row[q] = static_cast<scalar_t>(gather(t, q));
    }
  }
  }
}

// Merge low- and high-pass band pairs back along one spatial axis.
// `(batch, 2 * groups, ...)` in, `(batch, groups, ...)` out.
torch::Tensor idwt_axis_cpu(const torch::Tensor& coeffs,
                            const torch::Tensor& rec_lo,
                            const torch::Tensor& rec_hi, int64_t axis,
                            int64_t mode, int64_t trim, int64_t out_length,
                            c10::optional<torch::Tensor> out_opt) {
  C3LI_CHECK_CONTIGUOUS(coeffs);
  C3LI_CHECK_FLOATING(coeffs);
  TORCH_CHECK(coeffs.size(1) % 2 == 0,
              "the channel count must pair a low- and a high-pass band");

  const AxisLayout layout = axis_layout(coeffs, axis);
  const int64_t filter_len = rec_lo.numel();
  const int64_t bands = coeffs.size(1);
  const int64_t groups = bands / 2;
  const int64_t coeff_length = layout.length;
  const bool circular = mode == kPeriodization;
  const int64_t inner = layout.inner;

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  sizes[2 + axis] = out_length;
  // As in the decomposition: a caller merging one axis after another hands the
  // same storage back every time, and freshly mapped pages cost more to fault
  // in than the merge costs to run.
  torch::Tensor out;
  if (out_opt.has_value()) {
    out = out_opt.value();
    TORCH_CHECK(out.sizes().vec() == sizes,
                "the output tensor does not have the shape the merge writes");
    TORCH_CHECK(out.scalar_type() == coeffs.scalar_type(),
                "the output tensor does not have the type of the bands");
    C3LI_CHECK_CONTIGUOUS(out);
  } else {
    out = torch::empty(sizes, coeffs.options());
  }

  const int64_t per_group = layout.outer / (coeffs.size(0) * bands);
  const int64_t lanes = coeffs.size(0) * groups * per_group;

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_axis_cpu", [&] {
    const auto* src = coeffs.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = rec_lo.to(coeffs.scalar_type()).contiguous();
    const auto hi_filter = rec_hi.to(coeffs.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();

    parallel_for(lanes, [&](int64_t begin, int64_t end) {
      for (int64_t o = begin; o < end; ++o) {
        const int64_t pre = o % per_group;
        const int64_t group = (o / per_group) % groups;
        const int64_t batch = o / (per_group * groups);
        const int64_t low_pre = (batch * bands + 2 * group) * per_group + pre;
        const scalar_t* low = src + low_pre * coeff_length * inner;
        const scalar_t* high = low + per_group * coeff_length * inner;
        const int64_t out_pre = (batch * groups + group) * per_group + pre;
        idwt_plane(low, high, dst + out_pre * out_length * inner, lo, hi,
                   filter_len, coeff_length, out_length, inner, trim, circular);
      }
    });
  });
  return out;
}

// Merge every band back over every spatial axis in one pass.
// `(batch, 2**d * groups, ...)` in, `(batch, groups, ...)` out.
//
// The per-axis form writes a whole tensor between axes, so a two-dimensional
// reconstruction streams its result through memory three times over. Here one
// lane is carried through every axis in a scratch buffer that stays in cache,
// and only the finished lane is written out.
torch::Tensor idwt_nd_cpu(const torch::Tensor& coeffs,
                          const torch::Tensor& rec_lo,
                          const torch::Tensor& rec_hi, int64_t mode,
                          const std::vector<int64_t>& trims,
                          const std::vector<int64_t>& out_lengths) {
  C3LI_CHECK_CONTIGUOUS(coeffs);
  C3LI_CHECK_FLOATING(coeffs);
  const int64_t dimensions = static_cast<int64_t>(out_lengths.size());
  TORCH_CHECK(dimensions >= 1 && dimensions <= 3,
              "a fused reconstruction runs over one to three axes");
  TORCH_CHECK(static_cast<int64_t>(trims.size()) == dimensions,
              "every axis needs a trim");
  TORCH_CHECK(coeffs.dim() == dimensions + 2,
              "the bands must carry a batch and a channel axis");
  const int64_t corners = int64_t{1} << dimensions;
  const int64_t bands = coeffs.size(1);
  TORCH_CHECK(bands % corners == 0,
              "the channel count must carry every subband of every group");

  const int64_t groups = bands / corners;
  const int64_t batch = coeffs.size(0);
  const bool circular = mode == kPeriodization;
  const int64_t filter_len = rec_lo.numel();

  std::vector<int64_t> coeff_shape(dimensions);
  for (int64_t d = 0; d < dimensions; ++d) {
    coeff_shape[d] = coeffs.size(2 + d);
  }

  auto sizes = coeffs.sizes().vec();
  sizes[1] = groups;
  for (int64_t d = 0; d < dimensions; ++d) {
    sizes[2 + d] = out_lengths[d];
  }
  torch::Tensor out = torch::empty(sizes, coeffs.options());

  // the widest a stage gets: axes already done are output length, the rest
  // are still coefficient length
  int64_t scratch = 0;
  for (int64_t stage = 0; stage <= dimensions; ++stage) {
    int64_t planes = int64_t{1} << (dimensions - stage);
    int64_t extent = planes;
    for (int64_t d = 0; d < dimensions; ++d) {
      extent *= (d >= dimensions - stage) ? out_lengths[d] : coeff_shape[d];
    }
    scratch = std::max(scratch, extent);
  }

  C3LI_DISPATCH_FLOATING(coeffs.scalar_type(), "idwt_nd_cpu", [&] {
    const auto* src = coeffs.data_ptr<scalar_t>();
    auto* dst = out.data_ptr<scalar_t>();
    const auto lo_filter = rec_lo.to(coeffs.scalar_type()).contiguous();
    const auto hi_filter = rec_hi.to(coeffs.scalar_type()).contiguous();
    const auto* lo = lo_filter.data_ptr<scalar_t>();
    const auto* hi = hi_filter.data_ptr<scalar_t>();

    int64_t coeff_lane = 1;
    for (int64_t d = 0; d < dimensions; ++d) {
      coeff_lane *= coeff_shape[d];
    }
    int64_t out_lane = 1;
    for (int64_t d = 0; d < dimensions; ++d) {
      out_lane *= out_lengths[d];
    }

    parallel_for(batch * groups, [&](int64_t begin, int64_t end) {
      // kept across calls: a lane's working set is small and reallocating it
      // per chunk costs more than the merge does on a short axis
      static thread_local std::vector<scalar_t> front_buf;
      static thread_local std::vector<scalar_t> back_buf;
      if (static_cast<int64_t>(front_buf.size()) < scratch) {
        front_buf.resize(scratch);
        back_buf.resize(scratch);
      }
      std::vector<scalar_t>& front = front_buf;
      std::vector<scalar_t>& back = back_buf;
      for (int64_t o = begin; o < end; ++o) {
        const int64_t group = o % groups;
        const int64_t b = o / groups;
        // the subbands of this group sit `groups` apart, the order the
        // decomposition stacked them in
        for (int64_t c = 0; c < corners; ++c) {
          const scalar_t* plane =
              src + ((b * bands) + c * groups + group) * coeff_lane;
          std::copy(plane, plane + coeff_lane, front.data() + c * coeff_lane);
        }

        // axes fold from the last inwards, halving the planes each time.
        // A plane still carries the axes not yet done, so the ones outside
        // this axis are walked here: the merge itself only ever sees a
        // `length` by `inner` slab.
        int64_t inner = 1;
        int64_t plane_len = coeff_lane;
        for (int64_t axis = dimensions - 1; axis >= 0; --axis) {
          const int64_t length = coeff_shape[axis];
          const int64_t wanted = out_lengths[axis];
          const int64_t outer = plane_len / (length * inner);
          const int64_t merged = outer * wanted * inner;
          const int64_t pairs = int64_t{1} << axis;
          for (int64_t pair = 0; pair < pairs; ++pair) {
            const scalar_t* low = front.data() + (2 * pair) * plane_len;
            const scalar_t* high = front.data() + (2 * pair + 1) * plane_len;
            scalar_t* merged_plane = back.data() + pair * merged;
            idwt_plane(low, high, merged_plane, lo, hi, filter_len, length,
                       wanted, inner, trims[axis], circular, outer);
          }
          front.swap(back);
          plane_len = merged;
          inner *= wanted;
        }
        std::copy(front.data(), front.data() + out_lane,
                  dst + (b * groups + group) * out_lane);
      }
    });
  });
  return out;
}

}  // namespace c3li
