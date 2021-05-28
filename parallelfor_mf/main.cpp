#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void test_setval (MultiFab& mf)
{
    double t_orig = 0., t_new = 0.;
    for (int itest = 0; itest < 2; ++itest) {
        double t0 = amrex::second();
        mf.setVal(0.33);
        double t1 = amrex::second();
        {
            auto ma = mf.arrays();
            amrex::ParallelFor(mf,
                               [=] AMREX_GPU_DEVICE (int box_no, int i, int j, int k)
            {
                ma[box_no](i,j,k) = 1.0;
            });
            Gpu::synchronize();
        }
        double t2 = amrex::second();
        t_orig = t1-t0;
        t_new = t2-t1;
    }

    amrex::Print() << "    Kernel run time is " << std::scientific << t_orig
                   << " " << t_new << ".\n";
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};
        MultiFab mf(ba, dm, 1, 0);

        test_setval(mf);
        if (mf.min(0) != 1.0 || mf.max(0) != 1.0) {
            amrex::Print() << "ERROR!!! setVal failed" << std::endl;
        }
    }
    amrex::Finalize();
}
