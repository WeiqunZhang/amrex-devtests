
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void main_main ()
{
    BL_PROFILE("main");

    int n_cell = 128;
    int max_grid_size = 16;
    std::string plot_file{"plt"};
    {
        ParmParse pp;
        pp.query("n_cell", n_cell);
        pp.query("max_grid_size", max_grid_size);
        pp.query("plot_file", plot_file);

        ParmParse ppeb2("eb2");
        std::string geom_type("stl");
        ppeb2.add("geom_type", geom_type);
        ppeb2.add("cover_multiple_cuts", 1);
        // ppeb2.add("stl_file", std::string("armadillo.stl"));
        ppeb2.add("stl_file", std::string("adirondack.stl"));
    }

    Geometry geom(Box(IntVect(0),IntVect(n_cell-1, n_cell-1, n_cell-1)),
#if 0
                  RealBox({AMREX_D_DECL(-100., -75., -100.)},
                          {AMREX_D_DECL( 100., 125.,  100.)}),
#else
                  RealBox({AMREX_D_DECL(   0.,   0.,   0.)},
                          {AMREX_D_DECL( 200., 200.,  50.)}),
#endif
                  0, {AMREX_D_DECL(0,0,0)});
    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid_size);
    DistributionMapping dm(ba);

    {
        BL_PROFILE("EB2:Build");
        EB2::Build(geom, 0, 0, 1);
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
