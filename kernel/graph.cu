


#include "nn_cuda.h"


#include <cmath>
#include <Eigen/Dense>

#include <open3d/core/Tensor.h>
#include <open3d/core/TensorKey.h>
#include <open3d/core/MemoryManager.h>
#include <open3d/t/geometry/kernel/GeometryIndexer.h>
#include <open3d/utility/Console.h>

#include "core/PlatformIndependence.h"
#include "geometry/WarpUtilities.h"





using namespace open3d;
namespace o3c = open3d::core;
namespace o3u = open3d::utility;
using namespace open3d::t::geometry::kernel;


static constexpr int64_t OPEN3D_PARFOR_BLOCK = 128;
static constexpr int64_t OPEN3D_PARFOR_THREAD = 4;

/// Calls f(n) with the "grid-stride loops" pattern.
template <int64_t block_size, int64_t thread_size, typename func_t>
__global__ void ElementWiseKernel_(int64_t n, func_t f) {
    int64_t items_per_block = block_size * thread_size;
    int64_t idx = blockIdx.x * items_per_block + threadIdx.x;
#pragma unroll
    for (int64_t i = 0; i < thread_size; ++i) {
        if (idx < n) {
            f(idx);
            idx += block_size;
        }
    }
}

/// Run a function in parallel on CUDA.
template <typename func_t>
void ParallelForCUDA_(const o3c::Device& device, int64_t n, const func_t& func) {
    if (device.GetType() != o3c::Device::DeviceType::CUDA) {
        o3u::LogError("ParallelFor for CUDA cannot run on device {}.",
                          device.ToString());
    }
    if (n == 0) {
        return;
    }

    // CUDAScopedDevice scoped_device(device);
    int64_t items_per_block = OPEN3D_PARFOR_BLOCK * OPEN3D_PARFOR_THREAD;
    int64_t grid_size = (n + items_per_block - 1) / items_per_block;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    ElementWiseKernel_<OPEN3D_PARFOR_BLOCK, OPEN3D_PARFOR_THREAD>
            <<<grid_size, OPEN3D_PARFOR_BLOCK, 0, stream>>>(
                    n, func);
    OPEN3D_GET_LAST_CUDA_ERROR("ParallelFor failed.");
}


template <typename func_t>
void ParallelFor(const o3c::Device& device, int64_t n, const func_t& func) {
// #ifdef __CUDACC__
    ParallelForCUDA_(device, n, func);
// #else
//     ParallelForCPU_(device, n, func);
// #endif
}


// // ================================================================//



using namespace nnrt::geometry::kernel::warp;

template<open3d::core::Device::DeviceType TDeviceType, bool TUseValidAnchorThreshold>
void ComputeAnchorsAndWeightsEuclidean
		(open3d::core::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points,
		 const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
		 const float node_coverage) {

	float node_coverage_squared = node_coverage * node_coverage;

	int64_t point_count = points.GetLength();
	int64_t node_count = nodes.GetLength();
	anchors = o3c::Tensor::Ones({point_count, anchor_count}, o3c::Dtype::Int32, nodes.GetDevice()) * -1;
	weights = o3c::Tensor({point_count, anchor_count}, o3c::Dtype::Float32, nodes.GetDevice());

	//input indexers
	NDArrayIndexer point_indexer(points, 1);
	NDArrayIndexer node_indexer(nodes, 1);

	//output indexers
	NDArrayIndexer anchor_indexer(anchors, 1);
	NDArrayIndexer weight_indexer(weights, 1);

	ParallelFor(
			points.GetDevice(), point_count,
			[=] OPEN3D_DEVICE(int64_t workload_idx) {
				auto point_data = point_indexer.GetDataPtr<float>(workload_idx);
				Eigen::Vector3f point(point_data[0], point_data[1], point_data[2]);

				// region ===================== COMPUTE ANCHOR POINTS & WEIGHTS ================================
				auto anchor_indices = anchor_indexer.template GetDataPtr<int32_t>(workload_idx);
				auto anchor_weights = weight_indexer.template GetDataPtr<float>(workload_idx);
				if (TUseValidAnchorThreshold) {
					nnrt::geometry::kernel::warp::FindAnchorsAndWeightsForPointEuclidean_Threshold<TDeviceType>(anchor_indices, anchor_weights, anchor_count,
					                                                                     minimum_valid_anchor_count, node_count,
					                                                                     point, node_indexer, node_coverage_squared);
				} else {
					nnrt::geometry::kernel::warp::FindAnchorsAndWeightsForPointEuclidean<TDeviceType>(anchor_indices, anchor_weights, anchor_count, node_count,
					                                                           point, node_indexer, node_coverage_squared);
				}
				// endregion
			}
	);
}


// void ComputeAnchorsAndWeightsEuclidean_C(open3d::core::Tensor& anchors, open3d::core::Tensor& weights, const open3d::core::Tensor& points,
//                                        const open3d::core::Tensor& nodes, const int anchor_count, const int minimum_valid_anchor_count,
//                                        const float node_coverage) {
void ComputeAnchorsAndWeightsEuclidean_C() {
	// open3d::core::Device device = points.GetDevice();
	// open3d::core::Device::DeviceType device_type = device.GetType();
	// open3d::core::Device a= core::Device::DeviceType::CPU;
	
	using namespace std;
	std::cout<<"ok!"<<std::endl;
// 	switch (device_type) {
// 		case core::Device::DeviceType::CPU:
// 			if (minimum_valid_anchor_count > 0) {
// 				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CPU, true>(
// 						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 			} else {
// 				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CPU, false>(
// 						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 			}
// 			break;
// 		case core::Device::DeviceType::CUDA:
// // #ifdef BUILD_CUDA_MODULE
// 			if (minimum_valid_anchor_count > 0) {
// 				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CUDA, true>(
// 						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 			} else {
// 				ComputeAnchorsAndWeightsEuclidean<core::Device::DeviceType::CUDA, false>(
// 						anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 			}
// // #else
// // 			utility::LogError("Not compiled with CUDA, but CUDA device is used.");
// // #endif
// 			break;
// 		default:
			utility::LogError("Unimplemented device");
// 			break;
// 	}
}


// void ComputeAnchorsAndWeightsEuclidean(o3c::Tensor& anchors, o3c::Tensor& weights, const o3c::Tensor& points, const o3c::Tensor& nodes,
//                                        int anchor_count, int minimum_valid_anchor_count, float node_coverage) {
// 	auto device = points.GetDevice();
// 	o3c::AssertTensorDtype(points, o3c::Dtype::Float32);
// 	o3c::AssertTensorDtype(nodes, o3c::Dtype::Float32);
// 	o3c::AssertTensorDevice(nodes, device);
// 	if (minimum_valid_anchor_count > anchor_count) {
// 		o3u::LogError("minimum_valid_anchor_count (now, {}) has to be smaller than or equal to anchor_count, which is {}.",
// 		              minimum_valid_anchor_count, anchor_count);
// 	}
// 	if (anchor_count < 1) {
// 		o3u::LogError("anchor_count needs to be greater than one. Got: {}.", anchor_count);
// 	}
// 	auto points_shape = points.GetShape();
// 	if (points_shape.size() < 2 || points_shape.size() > 3) {
// 		o3u::LogError("`points` needs to have 2 or 3 dimensions. Got: {} dimensions.", points_shape.size());
// 	}
// 	o3c::Tensor points_array;
// 	enum PointMode {
// 		POINT_ARRAY, POINT_IMAGE
// 	};
// 	PointMode point_mode;
// 	if (points_shape.size() == 2) {
// 		o3c::AssertTensorShape(points, { o3u::nullopt, 3 });
// 		points_array = points;
// 		point_mode = POINT_ARRAY;
// 	} else {
// 		o3c::AssertTensorShape(points, { o3u::nullopt, o3u::nullopt, 3 });
// 		points_array = points.Reshape({-1, 3});
// 		point_mode = POINT_IMAGE;
// 	}
// 	ComputeAnchorsAndWeightsEuclidean(anchors, weights, points_array, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 	if(point_mode == POINT_IMAGE){
// 		anchors = anchors.Reshape({points_shape[0], points_shape[1], anchor_count});
// 		weights = weights.Reshape({points_shape[0], points_shape[1], anchor_count});
// 	}
// }



// py::tuple compute_anchors_and_weights_euclidean(const o3c::Tensor& points, const o3c::Tensor& nodes, int anchor_count, int minimum_valid_anchor_count,
//                                             float node_coverage) {
// 	o3c::Tensor anchors, weights;
// 	ComputeAnchorsAndWeightsEuclidean(anchors, weights, points, nodes, anchor_count, minimum_valid_anchor_count, node_coverage);
// 	return py::make_tuple(anchors, weights);
// }
