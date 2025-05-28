
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void main_main ()
{
    BL_PROFILE("main");

    int n_cell_x = 64;
    int n_cell_y = 8;
    int n_cell_z = 8;
    int max_grid_size = 32;
    int max_level = 2;
    std::string plot_file{"plt"};
    {
        ParmParse pp;
        pp.query("n_cell_x", n_cell_x);
        pp.query("n_cell_y", n_cell_y);
        pp.query("n_cell_z", n_cell_z);
        pp.query("max_grid_size", max_grid_size);
        pp.query("max_level", max_level);
        pp.query("plot_file", plot_file);

        ParmParse ppeb2("eb2");
        std::string geom_type("stl");
        ppeb2.add("geom_type", geom_type);
        ppeb2.add("cover_multiple_cuts", 1);
        ppeb2.add("stl_file", std::string("NREL_PhaseVI_blade.stl"));
    }

    Vector<Geometry> geom;
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        if (geom.empty()) {
            geom.push_back(Geometry(Box(IntVect(0),IntVect(n_cell_x-1,n_cell_y-1,n_cell_z-1)),
                                    RealBox({0.  , -0.49, -0.49},
                                            {7.68,  0.49,  0.49}),
                                    0, {0,0,0}));
        } else {
            geom.push_back(amrex::refine(geom.back(), 2));
        }
    }

    EB2::Build(geom.back(), max_level, max_level, 1, false);

    Vector<MultiFab> mf(max_level+1);
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        auto const& gm = geom[ilev];
        Box bx = gm.Domain();
        bx.grow(-ilev*2); // shrink the fine level box a little bit
        BoxArray ba(bx);
        ba.maxSize(max_grid_size);
        DistributionMapping dm{ba};

        auto const& factory = makeEBFabFactory(gm, ba, dm, {1,1,1}, EBSupport::full);
        MultiCutFab const& barea = factory->getBndryArea();
        MultiCutFab const& bnorm = factory->getBndryNormal();

        int ncomp = 4; // boundary area: 1, boundary normal: 3
        mf[ilev].define(ba,dm,ncomp,0,MFInfo{},*factory);
        mf[ilev].ParallelCopy(barea.ToMultiFab(0.0, -1.0), 0, 0, 1);
        mf[ilev].ParallelCopy(bnorm.ToMultiFab(-1.0, -1.0), 0, 1, 3);
    }

    // EB_WriteMultiLevelPlotfile will automatically save volume fraction
    EB_WriteMultiLevelPlotfile("plot", max_level+1, GetVecOfConstPtrs(mf),
                               {"barea", "nx", "ny", "nz"}, geom, 0.0,
                               Vector<int>(max_level+1,0),
                               Vector<IntVect>(max_level,IntVect(2)));
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
