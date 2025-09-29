#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main");
        int n_cell = 4095;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
        }

        Box domain(IntVect(0), IntVect(n_cell-1,n_cell-1,0));
        Periodicity period(IntVect(n_cell,n_cell,0));
        BoxArray ba(domain);
        DistributionMapping dm{ba};

        MultiFab mf(ba, dm, 3, IntVect(2,2,0));
        mf.setVal(1);
        mf.SumBoundary(0, 3, period);

        {
            BL_PROFILE_REGION("SUM_BOUNDARY");
            for (int i = 0; i < 2000; ++i) {
                mf.SumBoundary(0, 3, period);
            }
        }
    }
    amrex::Finalize();
}
