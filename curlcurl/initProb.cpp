
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProb ()
{
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        exact[idim].setVal(0.0);
        rhs  [idim].setVal(0.0);
    }

#if 0
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.growntilebox(1);
        auto rhsfab = rhs.array(mfi);
        auto solfab = solution.array(mfi);
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            actual_init_poisson(i,j,k,rhsfab,solfab,prob_lo,prob_hi,dx);
        });
    }
#endif
}
