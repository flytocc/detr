#include <torch/extension.h>
#include "csrc/poly_iou.h"


namespace poly {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("poly_iou", &poly_iou, "IoU for polygons");
}

} // namespace poly
