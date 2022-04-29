#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

using namespace amrex;

void build_sphere (Array<Real,AMREX_SPACEDIM> const& center,
                   Real radius, Geometry const& geom)
{
    EB2::SphereIF sf(radius, center, false);
    EB2::GeometryShop<EB2::SphereIF> gshop(sf);
    EB2::Build(gshop, geom, 0, 0, 2);
}

void main_main ()
{
    Box domain(IntVect(0), IntVect(127));
    BoxArray grids(domain);
    grids.maxSize(16);
    DistributionMapping dmap(grids);
    Geometry geom(domain, RealBox(AMREX_D_DECL(0._rt,0._rt,0._rt),
                                  AMREX_D_DECL(1.e-4_rt,1.e-4_rt,1.e-4_rt)),
                  CoordSys::cartesian, {AMREX_D_DECL(0,1,0)});

    Parser parser("-((x-0.00005)**2+(y-0.00005)**2-1e-05**2)");
    parser.registerVariables({"x","y","z"});

    EB2::ParserIF pif(parser.compile<3>());
    auto gshop = EB2::makeShop(pif, parser);
    EB2::Build(gshop, geom, 0, 10);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    main_main();
    amrex::Finalize();
}
