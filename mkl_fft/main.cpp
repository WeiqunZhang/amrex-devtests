#include <AMReX.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuComplex.H>
#include <AMReX_ParmParse.H>

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
    {
        ParmParse pp("amrex");
        pp.add("the_arena_is_managed", std::string("true"));
        pp.add("verbose", 0);
    }
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

        // rfab is a column major array with a size of 80 in each direction.
        // cfab is a column major array with a size of 80 except that the
        // size in the first direction is 40.

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
#elif defined(AMREX_USE_SYCL)
        using VedorFFTPlan = oneapi::mkl::dft::descriptor
            <oneapi::mkl::dft::precision::DOUBLE,oneapi::mkl::dft::domain::REAL> *;
        VedorFFTPlan plan;
        std::vector<std::int64_t> strides(ndim+1,0);
        if (ndim == 1) {
            strides[1] = 1;
            plan = new std::remove_pointer_t<VedorFFTPlan>(
                std::int64_t(rbox.length(0)));
        } else if (ndim == 2) {
            strides[2] = 1;
            strides[1] = rbox.length(0);
            plan = new std::remove_pointer_t<VedorFFTPlan>(
                {std::int64_t(rbox.length(1)),
                 std::int64_t(rbox.length(0))});
        } else {
            strides[3] = 1;
            strides[2] = rbox.length(0);
            strides[1] = rbox.length(0) * rbox.length(1);
            plan = new std::remove_pointer_t<VedorFFTPlan>(
                {std::int64_t(rbox.length(2)),
                 std::int64_t(rbox.length(1)),
                 std::int64_t(rbox.length(0))});
        }

        plan->set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                        DFTI_NOT_INPLACE);
        plan->set_value(oneapi::mkl::dft::config_param::FWD_STRIDES,strides.data());
//        plan->set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT,
//                        DFTI_CCE_FORMAT);
//        plan->set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
//                        DFTI_COMPLEX_COMPLEX);
        plan->commit(amrex::Gpu::Device::streamQueue());

        sycl::event r = oneapi::mkl::dft::compute_forward
            (*plan, rfab.dataPtr(),
             reinterpret_cast<std::complex<amrex::Real>*>(cfab.dataPtr()));
        r.wait();

        Gpu::streamSynchronize();
        delete plan;
#else
        static_assert(false, "todo");
#endif

        FArrayBox resultfab(cbox,2);
        {
            auto const& dst = resultfab.array();
            auto const& src = cfab.const_array();
            ParallelFor(cbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                dst(i,j,k,0) = src(i,j,k).real();
                dst(i,j,k,1) = src(i,j,k).imag();
            });
            Gpu::streamSynchronize();
        }

#if 1
        amrex::Print() << "dim = " << ndim << "\n"
                       << "  real min & max "
                       << resultfab.template min<RunOn::Device>(0)
                       << " " << resultfab.template max<RunOn::Device>(0) << "\n"
                       << "  imag min & max "
                       << resultfab.template min<RunOn::Device>(1)
                       << " " << resultfab.template max<RunOn::Device>(1) << "\n";

        resultfab(IntVect(2,0,0),1) = 0.0;

        amrex::Print() << "  After resetting imag at (2,0,0) to zero, imag & max "
                       << resultfab.template min<RunOn::Device>(1)
                       << " " << resultfab.template max<RunOn::Device>(1) << "\n";

        {
            std::ofstream ofs("rfab"+std::to_string(ndim));
            rfab.writeOn(ofs);
        }
        {
            std::ofstream ofs("cfab"+std::to_string(ndim));
            resultfab.writeOn(ofs);
        }

        // Backward
        FArrayBox rfab2(rfab.box());

#if defined(AMREX_USE_CUDA)
        cufftHandle plan2;
        if (ndim == 1) {
            cufftPlan1d(&plan, rbox.length(0), CUFFT_Z2D, 1);
        } else if (ndim == 2) {
            cufftPlan2d(&plan, rbox.length(1), rbox.length(0), CUFFT_Z2D);
        } else {
            cufftPlan3d(&plan, rbox.length(2), rbox.length(1), rbox.length(0), CUFFT_Z2D);
        }
        cufftSetStream(plan, amrex::Gpu::gpuStream());
        cufftExecZ2D(plan, reinterpret_cast<cufftDoubleComplex*>(cfab.dataPtr()),
                     rfab2.dataPtr());
        Gpu::streamSynchronize();
        cufftDestroy(plan);
#endif

        rfab2.template mult<RunOn::Device>(Real(1.0)/rbox.d_numPts());
        rfab2.template minus<RunOn::Device>(rfab);
        auto rr = rfab2.template minmax<RunOn::Device>();
        amrex::Print() << "  After bwd fft, min & max: " << rr.first
                       << " " << rr.second << std::endl;
        {
            std::ofstream ofs("after-rfab"+std::to_string(ndim));
            rfab2.writeOn(ofs);
        }

#endif
    }
    amrex::Finalize();
}
