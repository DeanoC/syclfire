//
// Created by Computer on 23/01/2020.
//

// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_cmath/scalar.h"
#include "al2o3_memory/memory.h"
#include "fire.hpp"
#include <random>

Fire::Fire(uint32_t width_, uint32_t height_) :
		width{width_},
		height{height_},
		doubleBufferIndex{0},
		hostIntensity{(float *) MEMORY_CALLOC(width_ * height_, sizeof(float))},
		hostNewData{(float*) MEMORY_CALLOC(height_, sizeof(float))},
		dataRange{height_, width_} {

	using namespace cl::sycl;

	intensity[0] = buffer<float, 2>{dataRange};
	intensity[1] = buffer<float, 2>{dataRange};
	newData = cl::sycl::buffer<float, 1>(cl::sycl::range<1>(height));
}

Fire::~Fire() {
	MEMORY_FREE(hostNewData);
	MEMORY_FREE(hostIntensity);
}

namespace {
struct UpdateTag1;
struct InitTag1;
}

void Fire::init(cl::sycl::queue &q) {
	using namespace cl::sycl;

	try {
		q.submit([&](handler &cgh) {
			auto ptr1 = intensity[0].get_access<access::mode::discard_write>(cgh);
			cgh.fill<float>(ptr1, 0.0f);
		});
	} catch (sycl::exception const &e) {
		LOGERROR("Caught synchronous SYCL exception: %s", e.what());
	}

}

void Fire::update(cl::sycl::queue &q) {
	using namespace cl::sycl;
	auto const ndr = nd_range<2>{dataRange, range<2>(16,32)};
	try {
		q.submit([&](handler &cgh) {
			auto newDataPtr = newData.get_access<access::mode::discard_write>(cgh);
			std::random_device r;
			std::default_random_engine e1(r());
			std::uniform_int_distribution<int> uniform_dist(64, 128);
			for (uint32_t i = 0u; i < height; ++i) {
				hostNewData[i] = (float) uniform_dist(e1);
			}
			cgh.copy(hostNewData, newDataPtr);
		});

		q.submit([&](handler &cgh) {
			auto imp = intensity[doubleBufferIndex].get_access<access::mode::read>(cgh);
			auto outp = intensity[doubleBufferIndex ^ 1].get_access<access::mode::discard_write>(cgh);
			auto newDataPtr = newData.get_access<access::mode::read>(cgh);

			cgh.parallel_for<UpdateTag1>(ndr, [=](nd_item<2> item) {
				id<2> const gid = item.get_global_id();
				float const c1 = 0.53f;
				float const c2 = 0.225f;

				int2 const upv = (int2) gid + int2(-1, -1);
				int2 const downv = (int2) gid + int2(+1, -1);
				id<2> up{(size_t) upv.x(), (size_t) upv.y()};
				id<2> down{(size_t) downv.x(), (size_t) downv.y()};

				if (gid[0] == 0) {
					up[0] = item.get_global_range()[0] - 1;
				}
				if (gid[0] == item.get_global_range()[0] - 1) {
					down[0] = 0;
				}

				if (gid[1] == 0) {
					outp[gid] = newDataPtr[gid[0]];
				} else {
					outp[gid] = (c1 * imp[gid + id<2>(0, -1)]) +
							(c2 * imp[up]) +
							(c2 * imp[down]);
				}
			});

		});

		updateDoneEvent = q.submit([&](handler &cgh) {
			auto src = intensity[doubleBufferIndex ^ 1].get_access<access::mode::read>(cgh);
			cgh.copy(src, hostIntensity);
			doubleBufferIndex ^= 1;
		});

	} catch (std::exception const &e) {
		LOGERROR("Caught synchronous SYCL exception: %s", e.what());
	}
}

void Fire::flushToHost() {
	using namespace cl::sycl;

	try {
		updateDoneEvent.wait_and_throw();
	} catch (std::exception const &e) {
		LOGERROR("Caught synchronous SYCL exception: %s", e.what());
	}

}
