
#include <AMReX.H>
#include <AMReX_EB2.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void main_main ()
{
    int n_cell = 128;
    int max_grid_size = 64;
    std::string plot_file{"plt"};
    {
        ParmParse pp;
        pp.query("n_cell", n_cell);
        pp.query("max_grid_size", max_grid_size);
        pp.query("plot_file", plot_file);

        ParmParse ppeb2("eb2");
        std::string geom_type("stl");
        ppeb2.add("geom_type", geom_type);
    }

    Geometry geom(Box(IntVect(0),IntVect(n_cell-1)),
                  RealBox({AMREX_D_DECL(-1.2_rt,-1.2_rt,-1.2_rt)},
                          {AMREX_D_DECL( 1.2_rt, 1.2_rt, 1.2_rt)}),
                  0, {AMREX_D_DECL(0,0,0)});
    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid_size);
    DistributionMapping dm(ba);

    EB2::Build(geom, 0, 10);

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
