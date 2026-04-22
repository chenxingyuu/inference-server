#pragma once
#include <cstdint>

namespace infer {

/// Returns true if any dimension in [dims, dims+nb_dims) is negative,
/// indicating a dynamic TensorRT shape.
///
/// Rules:
///  - nb_dims <= 0 → false  (tensor not found, or 0-rank; no dynamic dims)
///  - Callers must not pass nullptr when nb_dims > 0
///
/// No TensorRT dependency: usable in unit tests without GPU hardware.
inline bool hasDynamicDims(int nb_dims, const int32_t* dims) {
    for (int i = 0; i < nb_dims; ++i)
        if (dims[i] < 0) return true;
    return false;
}

} // namespace infer
