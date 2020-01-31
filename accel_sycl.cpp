//
// Created by Computer on 21/01/2020.
//

// License Summary: MIT see LICENSE file

#include "al2o3_platform/platform.h"
#include "al2o3_memory/memory.h"
#include "accel_sycl.hpp"
#include <CL/sycl.hpp>

auto exception_handler = [] (sycl::exception_list exceptions) {
	for (std::exception_ptr const& e : exceptions) {
		try {
			std::rethrow_exception(e);
		}
		catch( cl::sycl::exception const e) {
			LOGERROR("%s", e.what());
		} catch( std::exception const e) {
			LOGERROR("%s", e.what());
		}
	}
};

struct SyclCore {
	SyclCore(cl::sycl::device &dev) :
			device(dev),
			queue(device, exception_handler) {
	}

	cl::sycl::device device;
	cl::sycl::queue queue;
};

inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {
			{0x30, 192},
			{0x32, 192},
			{0x35, 192},
			{0x37, 192},
			{0x50, 128},
			{0x52, 128},
			{0x53, 128},
			{0x60, 64},
			{0x61, 128},
			{0x62, 128},
			{0x70, 64},
			{0x72, 64},
			{0x75, 64},
			{-1, -1}};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	LOGINFO("MapSMtoCores for SM %d.%d is undefined. Default to use %d Cores/SM\n",
					major, minor, nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

class selftestTag;

/* Classes can inherit from the device_selector class to allow users
 * to dictate the criteria for choosing a device from those that might be
 * present on a system. This example looks for a device with SPIR support
 * and prefers GPUs over CPUs. */
class custom_selector : public cl::sycl::device_selector {
public:
	custom_selector() : device_selector() {}

	/* The selection is performed via the () operator in the base
	 * selector class.This method will be called once per device in each
	 * platform. Note that all platforms are evaluated whenever there is
	 * a device selection. */
	int operator()(const cl::sycl::device &device) const override {
		using namespace cl::sycl;
		bool const isGPU = device.get_info<info::device::device_type>() == info::device_type::gpu;

		LOGINFO("-----");
		LOGINFO("%s - %s", device.get_info<info::device::vendor>().c_str(),
						device.get_info<info::device::name>().c_str());

		uint32_t cuCount = device.get_info<info::device::max_compute_units>();
		uint32_t coresPerCU = 1;
		uint32_t flopsPerCore = 8;

		if(device.is_host()) {
			return 1;
		}

		if(device.get_info<info::device::vendor_id>() == 0x10de) {
			auto bytes = size_t{0};
			clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, 0, nullptr, &bytes);
			auto major = cl_uint(0);
			clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV, bytes, &major, nullptr);
			clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, 0, nullptr, &bytes);
			auto minor = cl_uint(0);
			clGetDeviceInfo(device.get(), CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV, bytes, &minor, nullptr);
			LOGINFO("NVIDIA Compute Version: %d.%d", major, minor);
			coresPerCU = _ConvertSMVer2Cores(major, minor) / 32;
			flopsPerCore = 32;
		}

#if COMPUTECPP_BACKEND_spir64
		if( !device.supports_backend(detail::device_backend::SPIR) )
			return 0;
#endif
#if COMPUTECPP_BACKEND_spirv64
		if( !device.supports_backend(detail::device_backend::SPIRV) )
			return 0;
#endif
#if COMPUTECPP_BACKEND_ptx64
		if( !device.supports_backend(detail::device_backend::PTX) )
			return 0;
#endif

		LOGINFO("CUs - %d, cores per CU %d, total FALU %d gpu - %d",
				cuCount, coresPerCU, cuCount * coresPerCU * flopsPerCore, isGPU);

		return cuCount * coresPerCU * flopsPerCore;
	}

};

AL2O3_EXTERN_C SyclHandle AccelSycl_Create() {
	using namespace cl::sycl;

	custom_selector selector;
	device dev = selector.select_device();
	LOGINFO("Accelerator: %s %s has %d CUs @ %dMhz with %dKB local memory",
					dev.get_info<info::device::vendor>().c_str(),
					dev.get_info<info::device::name>().c_str(),
					dev.get_info<info::device::max_compute_units>(),
					dev.get_info<info::device::max_clock_frequency>(),

					(int) dev.get_info<info::device::local_mem_size>() / 1024);

	auto sycl = MEMORY_NEW(SyclCore, dev);

	const int dataSize = 128;
	float data[dataSize * dataSize] = {0.f};

	range<2> dataRange(dataSize,dataSize);
	buffer<float, 2> buf(data, dataRange);

	try {
		sycl->queue.submit([&](handler &cgh) {
			auto ptr = buf.get_access<access::mode::read_write>(cgh);

			cgh.parallel_for<selftestTag>(dataRange, [=](item<2> item) {
				size_t idx = item.get_id(0);
				ptr[item.get_id()] = static_cast<float>(idx);
			});
		});
	} catch( cl::sycl::exception const e) {
		LOGERROR("%s", e.what());
	} catch( std::exception const e) {
		LOGERROR("%s", e.what());
	}
	/* A host accessor can be used to force an update from the device to the
	 * host, allowing the data to be checked. */
	accessor<float, 2, access::mode::read, access::target::host_buffer>
			hostPtr(buf);

	if (hostPtr[id<2>(127,0)] != 127.0f) {
		LOGINFO("Sycl self test Failed");
		AccelSycl_Destroy(sycl);
		return nullptr;
	}

	return sycl;
}

AL2O3_EXTERN_C void AccelSycl_Destroy(SyclHandle sycl) {
	if(sycl) {
		LOGINFO("Destroying Sycl Accelerator");
		try {
			sycl->queue.wait_and_throw();
			MEMORY_DELETE(SyclCore, sycl);
		} catch( std::exception const e) {
			LOGERROR("Exception %s", e.what());
		}
	}
}
cl::sycl::queue& Accel::Sycl::getQueue() {
	return ((SyclCore*)this)->queue;
}

