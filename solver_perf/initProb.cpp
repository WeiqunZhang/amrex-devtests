
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbPoisson ()
{
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
}

void
MyTest::initProbABecLaplacian ()
{
    acoef.setVal(1.0);
    for (auto& bc : bcoef) {
        bc.setVal(1.0);
    }

    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
    auto a = ascalar;
    auto b = bscalar;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.growntilebox(1);
        auto solfab = solution.array(mfi);
        auto rhsfab = rhs.array(mfi);
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            actual_init_abeclap(i,j,k,rhsfab,solfab,a,b,prob_lo,prob_hi,dx);
        });
    }
}
