#include <cuda_runtime.h>
// Selects the SWITCH branch for this launch: index = clamp(width, 1, branches) - 1.
extern "C" __global__ void flare_select_conditional(cudaGraphConditionalHandle handle, const int *width, unsigned branches) {
  int w = *width;
  unsigned index = w < 1 ? 0u : (unsigned)w - 1u;
  cudaGraphSetConditional(handle, index < branches ? index : branches - 1u);
}
