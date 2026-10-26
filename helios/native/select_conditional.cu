// Built into native/select_conditional.fatbin:
//   nvcc -fatbin -gencode arch=compute_120,code=sm_120 -gencode arch=compute_100,code=sm_100 \
//        -gencode arch=compute_90,code=sm_90 -gencode arch=compute_80,code=compute_80
#include <cuda_runtime.h>
// Selects the SWITCH branch for this launch: index = clamp(width, 1, branches) - 1.
extern "C" __global__ void flare_select_conditional(cudaGraphConditionalHandle handle, const int *width, unsigned branches) {
  int w = *width;
  unsigned index = w < 1 ? 0u : (unsigned)w - 1u;
  cudaGraphSetConditional(handle, index < branches ? index : branches - 1u);
}
