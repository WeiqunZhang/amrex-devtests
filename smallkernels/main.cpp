#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        int n_cell=64, max_grid_size=64;
        ParmParse pp;
        pp.query("n_cell",        n_cell);
        pp.query("max_grid_size", max_grid_size);

        Box domain(IntVect(       0,        0,        0),
                   IntVect(n_cell-1, n_cell-1, n_cell-1));


        amrex::Print() << "domain: " << domain << "\n";

        BoxArray ba(domain);
        ba.maxSize(max_grid_size);

        DistributionMapping dm(ba);

        MultiFab mf0(ba, dm, 1, 1);
        MultiFab mf1(ba, dm, 1, 1);
        MultiFab mf2(ba, dm, 1, 1);

        mf0.setVal(0.1);
        mf1.setVal(0.2);
        mf2.setVal(0.3);

        int nRun = 100;

        Real tLinComb = 0.0, tMultiply = 0.0, tDot = 0.0;
        Real res      = 0.0;

        // warp up the cache
        for (int i=0; i<nRun; ++i) {
            MultiFab::Multiply(mf1, mf0, 0, 0, 1, 0);
            MultiFab::LinComb(mf2, 0.9999, mf0, 0, 0.9999, mf1, 0, 0, 1, 0);
            res += MultiFab::Dot(mf2, 0, 1, 0, true);
        }

        // actual timing
        for (int i=0; i<nRun; ++i) {
            auto t0 = amrex::second();
            MultiFab::Multiply(mf1, mf0, 0, 0, 1, 0);
            auto t1 = amrex::second();
            tMultiply += t1-t0;

            t0 = amrex::second();
            MultiFab::LinComb(mf2, 0.9999, mf0, 0, 0.9999, mf1, 0, 0, 1, 0);
            t1 = amrex::second();
            tLinComb += t1-t0;

            t0 = amrex::second();
            res += MultiFab::Dot(mf2, 0, 1, 0, true);
            t1 = amrex::second();
            tDot += t1-t0;
        }

        if (res > 0.0) {
            amrex::Print() << "Times: " << tLinComb << " " << tMultiply << " " << tDot << "\n";
        }
    }

    amrex::Finalize();
}
