#include <AMReX.H>
#include <AMReX_FFT.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

#if defined(USE_HEFFTE)
#include <heffte.h>
#endif

using namespace amrex;

#define USE_LOCAL_FFT 1

void test_amrex (Box const& domain, MultiFab& mf, cMultiFab& cmf)
{
    Gpu::synchronize();
    double ta = amrex::second();

#ifdef USE_LOCAL_FFT
    FFT::LocalR2C<Real,FFT::Direction::both> r2c(domain.length());
#else
    FFT::R2C<Real,FFT::Direction::both> r2c(domain);
#endif

    Gpu::synchronize();
    double tb = amrex::second();
#ifdef USE_LOCAL_FFT
    amrex::Print() << "cufft: Make plan time " << tb-ta << std::endl;
#else
    amrex::Print() << "amrex: Make plan time " << tb-ta << std::endl;
#endif

    for (int itest = 0; itest < 4; ++itest) {
        Gpu::synchronize();    
        double t0 = amrex::second();
#ifdef USE_LOCAL_FFT
        r2c.forward(mf[0].dataPtr(), cmf[0].dataPtr());
#else
        r2c.forward(mf, cmf);
#endif
        Gpu::synchronize();    
        double t1 = amrex::second();
#ifdef USE_LOCAL_FFT
        r2c.backward(cmf[0].dataPtr(), mf[0].dataPtr());
#else
        r2c.backward(cmf, mf);
#endif
        Gpu::synchronize();    
        double t2 = amrex::second();
        amrex::Print() << "Test # " << itest << ": forward time " << t1-t0 << ", backward time " << t2-t1 << std::endl;
    }
}

#ifdef USE_HEFFTE
void test_heffte (Box const& /*domain*/, MultiFab& mf, cMultiFab& cmf)
{
    auto& fab = mf[ParallelDescriptor::MyProc()];
    auto& cfab = cmf[ParallelDescriptor::MyProc()];

    auto const& local_box = fab.box();
    auto const& c_local_box = cfab.box();

    Gpu::synchronize();
    double ta = amrex::second();

#ifdef AMREX_USE_CUDA
    heffte::fft3d_r2c<heffte::backend::cufft> fft
#elif AMREX_USE_HIP
    heffte::fft3d_r2c<heffte::backend::rocfft> fft
#else
    heffte::fft3d_r2c<heffte::backend::fftw> fft
#endif
        ({{local_box.smallEnd(0),local_box.smallEnd(1),local_box.smallEnd(2)},
          {local_box.bigEnd(0)  ,local_box.bigEnd(1)  ,local_box.bigEnd(2)}},
         {{c_local_box.smallEnd(0),c_local_box.smallEnd(1),c_local_box.smallEnd(2)},
          {c_local_box.bigEnd(0)  ,c_local_box.bigEnd(1)  ,c_local_box.bigEnd(2)}},
         0, ParallelDescriptor::Communicator());

    using heffte_complex = typename heffte::fft_output<Real>::type;

    auto workspace_size = fft.size_workspace();
    Gpu::DeviceVector<heffte_complex> scratch(workspace_size);

    Gpu::synchronize();
    double tb = amrex::second();
    amrex::Print() << "heFFTe: Make plan time " << tb-ta << std::endl;

    for (int itest = 0; itest < 4; ++itest) {
        Gpu::synchronize();    
        double t0 = amrex::second();
        fft.forward(fab.dataPtr(), (heffte_complex*)cfab.dataPtr(),
                    scratch.data());
        Gpu::synchronize();    
        double t1 = amrex::second();
        fft.backward((heffte_complex*)cfab.dataPtr(), fab.dataPtr(),
                     scratch.data());
        Gpu::synchronize();    
        double t2 = amrex::second();
        amrex::Print() << "Test # " << itest << ": forward time " << t1-t0 << ", backward time " << t2-t1 << std::endl;
    }
}
#endif

int main (int argc, char* argv[])
{
    static_assert(AMREX_SPACEDIM == 3);

    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");

        AMREX_D_TERM(int n_cell_x = 256;,
                     int n_cell_y = 256;,
                     int n_cell_z = 256);

        {
            ParmParse pp;
            AMREX_D_TERM(pp.query("n_cell_x", n_cell_x);,
                         pp.query("n_cell_y", n_cell_y);,
                         pp.query("n_cell_z", n_cell_z));
        }

        amrex::Print() << "\n FFT size: " << n_cell_x << " " << n_cell_y << " " << n_cell_z << " "
                       << "  # of proc. " << ParallelDescriptor::NProcs() << "\n\n";

        Box domain(IntVect(0),IntVect(n_cell_x-1,n_cell_y-1,n_cell_z-1));
        BoxArray ba = amrex::decompose(domain, ParallelDescriptor::NProcs(), {true,true,true});
        AMREX_ALWAYS_ASSERT(ba.size() == ParallelDescriptor::NProcs());
        DistributionMapping dm = FFT::detail::make_iota_distromap(ba.size());

        GpuArray<Real,3> dx{1._rt/Real(n_cell_x), 1._rt/Real(n_cell_y), 1._rt/Real(n_cell_z)};

        MultiFab mf(ba, dm, 1, 0);
        auto const& ma = mf.arrays();
        ParallelFor(mf, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
        {
            AMREX_D_TERM(Real x = (i+0.5_rt) * dx[0] - 0.5_rt;,
                         Real y = (j+0.5_rt) * dx[1] - 0.5_rt;,
                         Real z = (k+0.5_rt) * dx[2] - 0.5_rt);
            ma[b](i,j,k) = std::exp(-10._rt*
                (AMREX_D_TERM(x*x*1.05_rt, + y*y*0.90_rt, + z*z)));
        });
        Gpu::streamSynchronize();

        Box cdomain(IntVect(0), IntVect(n_cell_x/2+1, n_cell_y-1, n_cell_z-1));
        BoxArray cba = amrex::decompose(cdomain, ParallelDescriptor::NProcs(), {true,true,true});
        AMREX_ALWAYS_ASSERT(cba.size() == ParallelDescriptor::NProcs());

        cMultiFab cmf(cba, dm, 1, 0);

#ifdef USE_HEFFTE
        test_heffte(domain, mf, cmf);
#else
        test_amrex(domain, mf, cmf);
#endif
    }
    amrex::Finalize();
}
