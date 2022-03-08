
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbALaplacian ()
{
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
    const int hd = 2;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(solution, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox();
        auto const& rhsfab = rhs.array(mfi);
        auto const& solfab = solution.array(mfi);
        auto const& acfab  = acoef.array(mfi);
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            actual_init_alaplacian(i,j,k,rhsfab,solfab,acfab,prob_lo,prob_hi,dx,hd);
        });
    }

    MultiFab::Copy(exact_solution, solution, 0, 0, 2, solution.nGrowVect());
    solution.setVal(0.0, 0, 2, 0);
}
