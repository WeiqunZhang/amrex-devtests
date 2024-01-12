#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

static void test_old (MultiFab& mf, MultiFab const& mf0)
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        Box tbx = mfi.fabbox();
        // Intentionally create potential race conditions
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        mf[mfi].template atomicAdd<RunOn::Device>(mf0[mfi], tbx, tbx, 0, 0, 1);
    }
}

static void test_new (MultiFab& mf, MultiFab const& mf0)
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        Box tbx = mfi.fabbox();
        // Intentionally create potential race conditions
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        mf[mfi].template lockAdd<RunOn::Device>(mf0[mfi], tbx, tbx, 0, 0, 1);
    }
}

static void test_race (MultiFab& mf, MultiFab const& mf0)
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        Box tbx = mfi.fabbox();
        // Intentionally create potential race conditions
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
        mf[mfi].template plus<RunOn::Device>(mf0[mfi], tbx, tbx, 0, 0, 1);
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncell = 128;
        int max_grid_size = 64;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            pp.query("max_grid_size", max_grid_size);
        }
        const int ng = 4;
        BoxArray ba(Box(IntVect(0),IntVect(ncell-1)));
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};
        MultiFab mf0(ba,dm,1,ng);
        MultiFab mf1(ba,dm,1,ng);
        MultiFab mf2(ba,dm,1,ng);
        MultiFab mf3(ba,dm,1,ng);
        amrex::FillRandom(mf0, 0, 1);
        mf1.setVal(0.0);
        mf2.setVal(0.0);
        mf3.setVal(0.0);

        const int ntests = 10;

        auto t0 = amrex::second();

        for (int i = 0; i < ntests; ++i) {
            test_old(mf2,mf0);
        }

        auto t1 = amrex::second();

        for (int i = 0; i < ntests; ++i) {
            test_new(mf1,mf0);
        }

        auto t2 = amrex::second();

        for (int i = 0; i < ntests; ++i) {
            test_race(mf3,mf0);
        }

        auto t3 = amrex::second();

        {
            MultiFab::Subtract(mf1, mf2, 0, 0, 1, ng);
            auto dmin = mf1.min(0, ng);
            auto dmax = mf1.max(0, ng);
            if (dmin != Real(0) || dmax != Real(0)) {
                amrex::AllPrint() << "  Test might have failed!!! " << dmin << " " << dmax << std::endl;
            }
        }

        {
            MultiFab::Subtract(mf3, mf2, 0, 0, 1, ng);
            auto dmin = mf3.min(0, ng);
            auto dmax = mf3.max(0, ng);
            if (dmin == Real(0) && dmax == Real(0)) {
                amrex::AllPrint() << "  Test has no race conditions?!" << std::endl;
            }
        }

        amrex::Print() << "     atomicAdd time: " << (t1-t0)/ntests << "\n"
                       << "       lockAdd time: " << (t2-t1)/ntests << "\n"
                       << "  nonatomicAdd time: " << (t3-t2)/ntests << "\n";
    }
    amrex::Finalize();
}
