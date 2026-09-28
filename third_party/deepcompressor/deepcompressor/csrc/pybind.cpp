#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "quantize/quantize.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("round_to_nearest_in_codebook_cuda", &round_to_nearest_in_codebook_cuda,
        py::arg("tensor"), py::arg("codebook"), py::arg("inplace") = false,
        py::arg("bnb") = false, "RTN with codebook (CUDA)");
}
