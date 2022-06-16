#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void test_setval (MultiFab& mf)
{
    double t_multli = 0., t_single = 0.;
    for (int itest = 0; itest < 2; ++itest) {
        double t0 = amrex::second();

        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
            auto const& a = mf.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                a(i,j,k) = 1.0;
            });
        }

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

        t_multli = t1-t0;
        t_single = t2-t1;
    }

    amrex::Print() << "Run time with " << mf.size() << " kernels is "
                   << std::scientific << t_multli << "\n"
                   << "Run time with fused kernel launch is "
                   << std::scientific << t_single << std::endl;;

}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
#ifdef AMREX_USE_MPI
    static_assert(false, "This is a serial test.");
#endif
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size = 64;
        std::vector<int> box_sizes;
        std::vector<int> nboxes;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            pp.query("max_grid_size", max_grid_size);
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
