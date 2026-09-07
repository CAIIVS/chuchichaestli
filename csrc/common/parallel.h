// SPDX-FileCopyrightText: 2024-present Members of CAIIVS
// SPDX-FileNotice: Part of chuchichaestli
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <ATen/Parallel.h>

namespace c3li {

// Items per task; below this the threading costs more than it saves.
constexpr int64_t kGrainSize = 2048;

// Run `body(begin, end)` over `[0, total)`, in parallel where it pays.
template <typename Body>
inline void parallel_for(int64_t total, const Body& body) {
  at::parallel_for(0, total, kGrainSize, body);
}

}  // namespace c3li
