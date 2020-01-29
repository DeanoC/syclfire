// License Summary: MIT see LICENSE file
#pragma once

namespace cl::sycl {
 class queue;
}

typedef struct SyclCore *SyclHandle;

AL2O3_EXTERN_C SyclHandle AccelSycl_Create();
AL2O3_EXTERN_C void AccelSycl_Destroy(SyclHandle handle);

#if defined(__cplusplus)
namespace Accel {

struct Sycl {
public:
	static Sycl* Create() {
		return (Sycl*)AccelSycl_Create();
	}

	void Destroy() {
		AccelSycl_Destroy((SyclHandle)this);
	}

	cl::sycl::queue& getQueue();

	~Sycl() {
		Destroy();
	}

	Sycl() = delete;
};

}
#endif