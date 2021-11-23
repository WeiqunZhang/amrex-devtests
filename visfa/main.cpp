#include <AMReX_VisMF.H>
#include <AMReX.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Random.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box box(IntVect(0), IntVect(127));
        BoxArray ba(box);
        ba.maxSize(64);
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
            amrex::Write(imf, "imf");
        }
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
            amrex::Write(mf, "mf");
        }
    }
    amrex::Finalize();
}
