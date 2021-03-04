
#include "MyTest.H"
#include "initProb_K.H"

using namespace amrex;

void
MyTest::initProbABecLaplacian ()
{
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
    const auto dlo     = amrex::lbound(geom.Domain());
    const auto dhi     = amrex::ubound(geom.Domain());
    const auto a = ascalar;
    const auto b = bscalar;
    const auto rdir = robin_dir;
    const auto rface = robin_face;
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        const Box& gbx = amrex::grow(bx,1);
        const auto& rhsfab = rhs.array(mfi);
        const auto& solfab = solution.array(mfi);
        const auto& acfab  = acoef.array(mfi);
        const auto& bcfab  = bcoef.array(mfi);
        const auto& rafab  = robin_a.array(mfi);
        const auto& rbfab  = robin_b.array(mfi);
        const auto& rffab  = robin_f.array(mfi);
        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            actual_init_abeclap(i,j,k,rhsfab,solfab,acfab,bcfab,rafab,rbfab,rffab,
                                a, b, prob_lo, prob_hi, dx, dlo, dhi, bx, rdir, rface);
        });
    }

    MultiFab::Copy(exact_solution, solution, 0, 0, 1, 0);
    solution.setVal(0.0, 0, 1, IntVect(0));
}
