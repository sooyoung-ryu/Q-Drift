
#include <assert.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <torch/extension.h>
#include <type_traits>

#include "quantize.h"

// The following code is adapted from the bitsandbytes library:
// https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/csrc/kernels.cu#L232
template <bool ret_val, typename float_t>
__device__ __forceinline__
    typename std::conditional<ret_val, float_t, int>::type
    bnb_nearest_neighbor(float_t x, float_t *codebook, const int C)
{
  int mid = (C >> 1) - 1;
  int hi = C - 1;
  int lo = 0;

  float_t lval = codebook[lo];
  float_t hval = codebook[hi];
  float_t mval = codebook[mid];

  for (int step = (C >> 2); step > 0; step >>= 1)
  {
    if (x > mval)
    {
      lo = mid;
      lval = mval;
      mid += step;
    }
    else
    {
      hi = mid;
      hval = mval;
      mid -= step;
    }
    mval = codebook[mid];
  }

  if (x > mval)
  {
    if constexpr (ret_val)
    {
      return (x - mval > hval - x) ? hval : mval;
    }
    else
    {
      return (x - mval > hval - x) ? hi : mid;
    }
  }
  else
  {
    if constexpr (ret_val)
    {
      return (x - lval < mval - x) ? lval : mval;
    }
    else
    {
      return (x - lval < mval - x) ? lo : mid;
    }
  }
}

template <bool ret_val, typename float_t>
__device__ __forceinline__
    typename std::conditional<ret_val, float_t, int>::type
    nearest_neighbor(float_t x, const float_t *codebook, int C)
{
  int lo = 0;
  int bit = 1 << (31 - __clz(C));

  float_t lval = codebook[lo];
  while (bit)
  {
    int next = lo | bit;
    float_t nval = codebook[next];
    bool pred = next < C && nval <= x;
    lo = pred ? next : lo;
    lval = pred ? nval : lval;
    bit >>= 1;
  }

  int hi = lo + (lo < C - 1);
  float_t hval = codebook[hi];

  if constexpr (ret_val)
  {
    return (x + x < lval + hval) ? lval : hval;
  }
  else
  {
    return (x + x < lval + hval) ? lo : hi;
  }
}

// CUDA kernel: Each thread processes one element from x and finds the nearest
// codebook entry. The codebook (of size C < 256) is first loaded into shared
// memory.
template <typename float_t, bool bnb = false>
__global__ void round_to_nearest_in_codebook_kernel(
    const float_t *__restrict__ x, const float_t *__restrict__ codebook,
    float_t *__restrict__ y, const int N, const int C)
{
  // Use a shared memory array for the codebook.
  __shared__ float_t s_codebook[256];

  // Have the first few threads load the codebook into shared memory.
  for (int i = threadIdx.x; i < C; i += blockDim.x)
  {
    s_codebook[i] = codebook[i];
  }
  __syncthreads();

  // Global index for the element processed by this thread.
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
  {
    if constexpr (bnb)
    {
      y[idx] = bnb_nearest_neighbor<true>(x[idx], s_codebook, C);
    }
    else
    {
      y[idx] = nearest_neighbor<true>(x[idx], s_codebook, C);
    }
  }
}

torch::Tensor round_to_nearest_in_codebook_cuda(torch::Tensor tensor,
                                                torch::Tensor codebook,
                                                bool inplace, bool bnb)
{
  auto x = tensor.contiguous();
  auto c = codebook.contiguous();
  auto y = inplace ? x : torch::empty_like(tensor);
  const int N = x.numel();
  const int C = c.numel();
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(
      tensor.scalar_type(), "round_to_nearest_in_codebook_cuda", [&]
      {
        if (bnb && (C & (C - 1)) == 0) {
          round_to_nearest_in_codebook_kernel<scalar_t, true>
              <<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                    c.data_ptr<scalar_t>(),
                                    y.data_ptr<scalar_t>(), N, C);
        } else {
          round_to_nearest_in_codebook_kernel<scalar_t, false>
              <<<blocks, threads>>>(x.data_ptr<scalar_t>(),
                                    c.data_ptr<scalar_t>(),
                                    y.data_ptr<scalar_t>(), N, C);
        } });
  return y;
}