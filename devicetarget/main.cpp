#include <iostream>

#if defined(AMREX_USE_CUDA)
#  include <cuda_runtime.h>
#elif defined(AMREX_USE_HIP)
#  include <hip/hip_runtime.h
#endif

#define AMREX_DEVICE_COMPILE (__CUDA_ARCH__ || __HIP_DEVICE_COMPILE__ || __SYCL_DEVICE_ONLY__)

namespace amrex { namespace Target {

    __host__ __device__
    constexpr bool isDevice () {
#if AMREX_DEVICE_COMPILE
        return true;
#else
        return false;
#endif
    }
}}

template <class L>
__global__ void launch_global (L f) { f(); }

__host__ __device__ void f ()
{
    if constexpr (amrex::Target::isDevice()) {
        printf("On device\n");
    } else {
        std::cout << "On host\n";
    }
}

int main (int argc, char* argv[])
{
    launch_global<<<1,1>>>([=] __device__
    {
        f();
    });
}
