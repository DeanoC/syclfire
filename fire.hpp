// License Summary: MIT see LICENSE file
#pragma once

#include "al2o3_platform/platform.h"
#include <CL/sycl.hpp>

struct Fire {
	Fire(uint32_t width, uint32_t height);
	~Fire();

	void flushToHost();

	void init(cl::sycl::queue& q);
	void update(cl::sycl::queue& q);

	uint32_t width;
	uint32_t height;

	uint32_t doubleBufferIndex;
	cl::sycl::range<2> dataRange;
	cl::sycl::range<2> downSampleRange;

	cl::sycl::event updateDoneEvent;


	float* hostNewData;
	cl::sycl::buffer<float, 1> newData;

	float* hostIntensity;
	cl::sycl::buffer<float, 2> intensity[2];
	cl::sycl::buffer<float, 2> downSample;
};


