#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void test (MultiFab& mf1, MultiFab& mf2, MultiFab& mf3)
{
    double t = 0.;
    for (int itest = 0; itest < 2; ++itest) {
        double t0 = amrex::second();
        MultiFab::Copy(mf1, mf2, 0, 0, 1, 0);
        double t1 = amrex::second();
        t = t1-t0;
    }
    amrex::Print() << "    Kernel run time is " << std::scientific << t << ".\n";
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
        MultiFab mf1(ba, dm, 1, 0);
        MultiFab mf2(ba, dm, 1, 0);
        MultiFab mf3(ba, dm, 1, 0);
        mf1.setVal(1.);
        mf2.setVal(2.);
        mf3.setVal(3.);

        test(mf1, mf2, mf3);
    }
    amrex::Finalize();
}
