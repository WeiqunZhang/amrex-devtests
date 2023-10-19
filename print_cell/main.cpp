#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        BL_PROFILE("main()");

        int ncell = 256;
        int max_grid_size;
        int ngrow = 1;
        std::vector<int> box_sizes;
        std::vector<int> nboxes;
        {
            ParmParse pp;
            pp.query("ncell", ncell);
            max_grid_size = ncell;
            pp.query("max_grid_size", max_grid_size);
            pp.query("ngrow", ngrow);
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
        MultiFab mf(ba, dm, 8, ngrow);
        FillRandom(mf, 0, mf.nComp());

        for (int icomp = 0; icomp < mf.nComp(); ++icomp) {
            printCell(mf, IntVect(128,128,128), icomp, IntVect(1));
        }
        printCell(mf, IntVect(128,128,128), -1, IntVect(1));
    }
    amrex::Finalize();
}
