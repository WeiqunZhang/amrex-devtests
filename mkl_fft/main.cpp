#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuComplex.H>

#if defined(AMREX_USE_CUDA)
#    include <cufft.h>
#elif defined(AMREX_USE_HIP)
#    if __has_include(<rocfft/rocfft.h>)  // ROCm 5.3+
#        include <rocfft/rocfft.h>
#    else
#        include <rocfft.h>
#    endif
#elif defined(AMREX_USE_SYCL)
#    include <oneapi/mkl/dfti.hpp>
#else
#    include <fftw3.h>
#endif

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    static_assert(std::is_same_v<Real,double>);

    int const n_cell = 80;
    Real dx = amrex::Math::pi<Real>() * Real(4.) / Real(n_cell);

    for (int ndim = 1; ndim <= 3; ++ndim)
    {
        IntVect hi(0);
        for (int idim = 0; idim < ndim; ++idim) { hi[idim] = n_cell-1; };
        Box rbox(IntVect(0), hi);
        FArrayBox rfab(rbox);

        for (int idim = 0; idim < ndim; ++idim) { hi[idim] = n_cell/2; };
        Box cbox(IntVect(0), hi);
        BaseFab<GpuComplex<Real>> cfab(cbox);

        auto const& ra = rfab.array();
        ParallelFor(rbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            ra(i,j,k) = std::sin(i*dx);
        });
        Gpu::streamSynchronize();

#if defined(AMREX_USE_CUDA)
        cufftHandle plan;
        if (ndim == 1) {
            cufftPlan1d(&plan, rbox.length(0), CUFFT_D2Z, 1);
        } else if (ndim == 2) {
            cufftPlan2d(&plan, rbox.length(1), rbox.length(0), CUFFT_D2Z);
        } else {
            cufftPlan3d(&plan, rbox.length(2), rbox.length(1), rbox.length(0), CUFFT_D2Z);
        }
        cufftSetStream(plan, amrex::Gpu::gpuStream());
        cufftExecD2Z(plan, rfab.dataPtr(),
                     reinterpret_cast<cufftDoubleComplex*>(cfab.dataPtr()));
        Gpu::streamSynchronize();
        cufftDestroy(plan);
#else
        static_assert(false, "todo");
#endif

        {
            std::ofstream ofs("rfab"+std::to_string(ndim));
            rfab.writeOn(ofs);
        }
        {
            std::ofstream ofs("cfab"+std::to_string(ndim));
            FArrayBox tmpfab(cbox,2);
            auto const& dst = tmpfab.array();
            auto const& src = cfab.const_array();
            ParallelFor(cbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dst(i,j,k,0) = src(i,j,k).real();
                dst(i,j,k,1) = src(i,j,k).imag();
            });
            Gpu::streamSynchronize();
            tmpfab.writeOn(ofs);
        }
    }
    amrex::Finalize();
}
