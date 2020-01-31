// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_cmath/scalar.h"
#include "al2o3_memory/memory.h"
#include "fire.hpp"
#include <random>

static int const SuperSampleRate = 8;

Fire::Fire(uint32_t width_, uint32_t height_) :
		width{width_},
		height{height_},
		doubleBufferIndex{0},
		hostIntensity{(float *) MEMORY_CALLOC(width_ * height_, sizeof(float))},
		hostNewData{(float *) MEMORY_CALLOC(height_ * SuperSampleRate, sizeof(float))},
		dataRange{height_ * SuperSampleRate, width_ * SuperSampleRate},
		downSampleRange{ height_, width_ }
{
	using namespace cl::sycl;

	LOGINFO("Fire size = %d x %d", dataRange[1], dataRange[0]);

	intensity[0] = buffer<float, 2>{dataRange};
	intensity[1] = buffer<float, 2>{dataRange};
	newData = buffer<float, 1>(cl::sycl::range<1>(dataRange[0] ));
	downSample = buffer<float, 2>{ downSampleRange };
}

Fire::~Fire() {
	MEMORY_FREE(hostNewData);
	MEMORY_FREE(hostIntensity);
}

namespace {
struct UpdateTag1;
struct DownSamplerTag1;
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
	auto const ndr = nd_range<2>{dataRange, range<2>(32, 32)};
	auto const dsndr = nd_range<2>{downSampleRange, range<2>(16, 32)};

	try {
		q.submit([&](handler &cgh) {
			auto newDataPtr = newData.get_access<access::mode::discard_write>(cgh);
			std::random_device r;
			std::default_random_engine e1(r());
			std::uniform_int_distribution<int> uniform_dist(64, 128);
			for (uint32_t i = 0u; i < dataRange[0]; ++i) {
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
				float const c1 = 0.497f;
				float const c2 = 0.251f;

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

		q.submit([&](handler &cgh) {
			auto src = intensity[doubleBufferIndex ^ 1].get_access<access::mode::read>(cgh);
			auto dst = downSample.get_access<access::mode::discard_write>(cgh);
			cgh.parallel_for<DownSamplerTag1>(dsndr, [=](nd_item<2> item) {
				id<2> gid = item.get_global_id();
				float accum = 0;
				for (int i = 0; i < SuperSampleRate; ++i) {
					id<2> agid;
					agid[0] = (gid[0] * SuperSampleRate) + i;
					agid[1] = (gid[1] * SuperSampleRate);
					for (int j = 0; j < SuperSampleRate; ++j) {
						accum += src[agid];
						agid[1] += 1;
					}
				}
				dst[gid] = accum * (1.0f / (SuperSampleRate*SuperSampleRate));
			});
			doubleBufferIndex ^= 1;
		});


		updateDoneEvent = q.submit([&](handler &cgh) {
			auto src = downSample.get_access<access::mode::read>(cgh);
			cgh.copy(src, hostIntensity);
		});

	} catch( cl::sycl::exception const e) {
		LOGERROR("%s", e.what());
	} catch( std::exception const e) {
		LOGERROR("%s", e.what());
	}
}

void Fire::flushToHost() {
	using namespace cl::sycl;

	try {
		updateDoneEvent.wait_and_throw();
	} catch( cl::sycl::exception const e) {
		LOGERROR("%s", e.what());
	} catch( std::exception const e) {
		LOGERROR("%s", e.what());
	}

}
