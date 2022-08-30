#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <eigen3/Eigen/Dense>
#include <open3d/core/Tensor.h>
#include <open3d/core/TensorKey.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/utility/Console.h>
namespace o3c = open3d::core;
namespace o3u = open3d::utility;

namespace py = pybind11;

void launch_add2(float *c,
                 const float *a,
                 const float *b,
                 int n);
void backproject_depth_float(py::array_t<float>& image_in, py::array_t<float>& point_image_out,
                             float fx, float fy, float cx, float cy);




template <typename func_t>
void ParallelFor_this(const o3c::Device& device, int64_t n, const func_t& func);


template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
void ComputeAnchorsAndWeightsEuclidean_cpp(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
                                       int anchor_count, int minimum_valid_anchor_count, float node_coverage);

void ComputeAnchorsAndWeightsEuclidean_C(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
                                       const open3d::core::Tensor& nodes, const int anchor_count, const int minimum_valid_anchor_count,
                                       const float node_coverage);

// void ComputeAnchorsAndWeightsEuclidean_C();
py::tuple ComputeAnchorsAndWeightsEuclidean_CCC(const open3d::core::Tensor& points, const open3d::core::Tensor& nodes, int anchor_count,
                                            int minimum_valid_anchor_count, float node_coverage);


// py::tuple compute_anchors_and_weights_euclidean(const o3c::Tensor& points, const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
//                                             float node_coverage);               


