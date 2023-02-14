#include <init.H>
#include <par_def.H>
#include <AMReX_Gpu.H>

#if defined(AMREX_USE_CUDA)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) cudaMemcpyToSymbol(d, h, n);
#elif defined(AMREX_USE_HIP)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) hipMemcpyToSymbol(d, h, n);
#elif defined(AMREX_USE_SYCL)
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) 
#else
#  define AMREX_GPU_MEMCPY_TO_SYMBOL(d,h,n) std::memcpy(&d, h, n);
#endif

void init ()
{
    int h_a = 3;
    amrex::GpuArray<int,4> h_b{10,20,30,40};

    AMREX_GPU_MEMCPY_TO_SYMBOL(d_a, &h_a, sizeof(int));
    AMREX_GPU_MEMCPY_TO_SYMBOL(d_b, h_b.data(), sizeof(int)*4);

    m_a = -3;
    m_b[0] = -10;
    m_b[1] = -20;
    m_b[2] = -30;
    m_b[3] = -40;

    amrex::Gpu::synchronize();
}
