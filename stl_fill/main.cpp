
#include <AMReX.H>
#include <AMReX_EB_STL_utils.H>
#include <AMReX_VisMF.H>

using namespace amrex;

void main_main ()
{
    int n_cell = 128;
    int max_grid_size = 64;
    std::string plot_file{"plt"};

    Geometry geom(Box(IntVect(0),IntVect(n_cell-1)),
                  RealBox({AMREX_D_DECL(-1.2_rt,-1.2_rt,-1.2_rt)},
                          {AMREX_D_DECL( 1.2_rt, 1.2_rt, 1.2_rt)}),
                  0, {AMREX_D_DECL(0,0,0)});
    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid_size);
    DistributionMapping dm(ba);
    MultiFab mf(ba,dm,1,0);

    std::string stl_file("sphere.stl");
    Real stl_scale = 1.0; // Scaling factor
    Array<Real,3> stl_center = {0.0, 0.0, 0.0}; // can be used to shift the object
    int stl_reverse_normal = false; // flip inside/outside

    STLtools stl_tools;
    stl_tools.read_stl_file(stl_file, stl_scale, stl_center, stl_reverse_normal);

    Real outside_value = 10.;
    Real inside_value = 100.;
    stl_tools.fill(mf, mf.nGrowVect(), geom, outside_value, inside_value);

    VisMF::Write(mf, "mf");
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
