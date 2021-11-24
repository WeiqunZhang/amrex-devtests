#include <AMReX_VisMF.H>
#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>

using namespace amrex;

template <typename MF>
void test (MF const& mf, std::string const& name)
{
    amrex::Write(mf, name);

    MF mf2;
    amrex::Read(mf2, name);

    MF mf3(mf.boxArray(), mf.DistributionMap(), mf.nComp(), mf.nGrowVect());
    mf3.Redistribute(mf2, 0, 0, mf.nComp(), mf.nGrowVect());

    MF::Subtract(mf3, mf, 0, 0, mf.nComp(), 1);
    auto imin = mf3.min(0, 1);
    auto imax = mf3.max(0, 1);
    if (imin != 0 || imax != 0) {
        amrex::Abort("amrex::Read(empty MF) failed");
    }

    amrex::Read(mf3, name);
    MF::Subtract(mf3, mf, 0, 0, mf.nComp(), 1);
    imin = mf3.min(0, 1);
    imax = mf3.max(0, 1);
    if (imin != 0 || imax != 0) {
        amrex::Abort("amrex::Read(defined MF) failed");
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    int n_cell = 128;
    int max_grid_size = 64;
    {
        ParmParse pp;
        pp.query("n_cell", n_cell);
        pp.query("max_grid_size", max_grid_size);
    }
    {
        Box box(IntVect(0), IntVect(n_cell-1));
        BoxArray ba(box);
        ba.maxSize(max_grid_size);
        DistributionMapping dm(ba);
        {
            iMultiFab imf(ba, dm, 2, IntVect(AMREX_D_DECL(1,2,1)));
            for (MFIter mfi(imf); mfi.isValid(); ++mfi) {
                auto const& fab = imf.array(mfi);
                amrex::ParallelForRNG(mfi.fabbox(), fab.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, RandomEngine const& eng)
                {
                    fab(i,j,k,n) = amrex::Random_int(1000000, eng);
                });
            }
            test(imf,"imf");
        }
#if 0
        {
            MultiFab mf(ba, dm, 2, IntVect(AMREX_D_DECL(1,2,1)));
            for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
                auto const& fab = mf.array(mfi);
                amrex::ParallelForRNG(mfi.fabbox(), fab.nComp(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, RandomEngine const& eng)
                {
                    fab(i,j,k,n) = amrex::Random(eng);
                });
            }
            test(mf, "mf");
        }
#endif
    }
    amrex::Finalize();
}
