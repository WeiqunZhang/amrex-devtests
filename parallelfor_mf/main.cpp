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

    if (mf.min(0) != 1.0 || mf.max(0) != 1.0) {
        amrex::Print() << "ERROR!!! setVal failed "
                       << mf.min(0) << ", " << mf.max(0) << std::endl;
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
        std::vector<int> box_sizes;
        std::vector<int> nboxes;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.queryarr("box_sizes", box_sizes);
            nboxes.resize(box_sizes.size(),1);
            pp.queryarr("nboxes", nboxes);
        }

        Box domain(IntVect(0),IntVect(ncell-1));
        BoxArray ba;
        if (box_sizes.empty()) {
            ba = BoxArray(domain);
            ba.maxSize(max_grid_size);
        } else {
            BoxList bl;
            for (int i = 0; i < box_sizes.size(); ++i) {
                for (int j = 0; j < nboxes[i]; ++j) {
                    bl.push_back(Box(IntVect(0), IntVect(box_sizes[i]-1)));
                }
            }
            ba = BoxArray(std::move(bl));
        }
        DistributionMapping dm{ba};
        MultiFab mf(ba, dm, 1, 0);

        test_setval(mf);
    }
    amrex::Finalize();
}
