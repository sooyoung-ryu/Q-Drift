#pragma once
#include <torch/extension.h>

torch::Tensor round_to_nearest_in_codebook_cuda(torch::Tensor tensor,
                                                torch::Tensor codebook,
                                                bool inplace = false,
                                                bool bnb = false);
