#include <torch/extension.h>
#include "nn_cuda.h"


// #include <eigen3/Eigen/Dense>

// #include <open3d/core/Tensor.h>
// #include <open3d/core/TensorKey.h>
// #include <open3d/core/MemoryManager.h>
// #include <open3d/t/geometry/kernel/GeometryIndexer.h>
// #include <open3d/utility/Console.h>


// #include "core/kernel/KnnUtilities.h"

// #include "geometry/WarpUtilities.h"

// using namespace nnrt::geometry::kernel::warp;
// using namespace nnrt::core::kernel::knn;

// using namespace open3d;
// namespace o3c = open3d::core;
// namespace o3u = open3d::utility;
// using namespace open3d::t::geometry::kernel;

// template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
// void ComputeAnchorsAndWeightsEuclidean
// 		(open3d::core::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points,
// 		 const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
// 		 const float node_coverage) {

// 	float node_coverage_squared = node_coverage * node_coverage;

// 	int64_t point_count = points.GetLength();
// 	int64_t node_count = nodes.GetLength();
// 	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
// 	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

// 	//input indexers
// 	NDArrayIndexer point_indexer(points, 1);
// 	NDArrayIndexer node_indexer(nodes, 1);

// 	//output indexers
// 	NDArrayIndexer anchor_indexer(anchors, 1);
// 	NDArrayIndexer weight_indexer(weights, 1);

// 	ParallelFor(
// 			points.GetDevice(), point_count,
// 			[=] OPEN3D_DEVICE(int64_t workload_idx) {
// 				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
// 				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

// 				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
// 				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
// 				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);
// 				if (TUseValidAnchorThreshold) {
// 					nnrt::geometry::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
// 					                                                                     minimum_valid_anchor_count, node_count,
// 					                                                                     point, node_indexer, node_coverage_squared);
// 				} else {
// 					nnrt::geometry::kernel::warp::FindAnchorsAndWeightsForPointEuclidean<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
// 					                                                           point, node_indexer, node_coverage_squared);
// 				}
// 				// endregion
// 			}
// 	);
// }


void torch_launch_add2(torch::Tensor &c,
                       const torch::Tensor &a,
                       const torch::Tensor &b,
                       int64_t n) {
    launch_add2((float *)c.data_ptr(),
                (const float *)a.data_ptr(),
                (const float *)b.data_ptr(),
                n);
}

void compute_anchors_and_weights_euclidean_c(){

      
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("torch_launch_add2",
            &torch_launch_add2,
            "add2 kernel warpper");
      m.def("launch_add2",
            &launch_add2,
            "add2 kernel warpper");
      m.def("backproject_depth_float",
            &backproject_depth_float,
            "backproject_depth_float");
      m.def("ComputeAnchorsAndWeightsEuclidean_C",
       &ComputeAnchorsAndWeightsEuclidean_C,
      "ComputeAnchorsAndWeightsEuclidean_C");   

	// m.def("compute_anchors_and_weights_euclidean",
	// 	&compute_anchors_and_weights_euclidean,
	// 	"compute_anchors_and_weights_euclidean");
      
      // m.def("compute_anchors_and_weights_euclidean", py::overload_cast<const o3c::Tensor&, const o3c::Tensor&, int, int,
      //                   float>(&compute_anchors_and_weights_euclidean), "points"_a, "nodes"_a, "anchor_count"_a,
      //       "minimum_valid_anchor_count"_a, "node_coverage"_a);
}

TORCH_LIBRARY(nn_cuda, m) {
    m.def("torch_launch_add2", torch_launch_add2);
}
