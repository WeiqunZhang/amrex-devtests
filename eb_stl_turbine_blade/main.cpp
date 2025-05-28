
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void main_main ()
{
    BL_PROFILE("main");

    int n_cell_x = 256;
    int n_cell_y = 32;
    int n_cell_z = 32;
    int max_grid_size = 32;
    std::string plot_file{"plt"};
    {
        ParmParse pp;
        pp.query("n_cell_x", n_cell_x);
        pp.query("n_cell_y", n_cell_y);
        pp.query("n_cell_z", n_cell_z);
        pp.query("max_grid_size", max_grid_size);
        pp.query("plot_file", plot_file);

        ParmParse ppeb2("eb2");
        std::string geom_type("stl");
        ppeb2.add("geom_type", geom_type);
        ppeb2.add("cover_multiple_cuts", 1);
        ppeb2.add("stl_file", std::string("NREL_PhaseVI_blade.stl"));
    }

    Geometry geom(Box(IntVect(0),IntVect(n_cell_x-1, n_cell_y-1, n_cell_z-1)),
                  RealBox({AMREX_D_DECL( 0.  , -0.49, -0.49)},
                          {AMREX_D_DECL( 7.68,  0.49,  0.49)}),
                  0, {AMREX_D_DECL(0,0,0)});
    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid_size);
    DistributionMapping dm(ba);

    {
        BL_PROFILE("EB2:Build");
        EB2::Build(geom, 0, 10, 1);
	amrex::Print() << "max coarsening level is " << EB2::maxCoarseningLevel(geom) << "\n";
    }

    auto const& factory = makeEBFabFactory(geom, ba, dm, {1,1,1}, EBSupport::full);
    MultiFab const& vfrc = factory->getVolFrac();
    amrex::WriteMLMF(plot_file, {&vfrc}, {geom});
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
