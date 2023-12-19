#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>
#include <AMReX_MLCurlCurl.H>

class MyTest
{
public:

    MyTest ();

    void solve ();

public: // make it public for cuda
    void initProb ();

private:

    void readParameters ();
    void initData ();

    std::array<int,AMREX_SPACEDIM> n_cells{AMREX_D_DECL(128,128,128)};
    int max_grid_size = 64;

    // For MLMG solver
    int verbose = 1;
    int bottom_verbose = 0;
    int max_iter = 100;
    bool agglomeration = true;
    bool consolidation = true;
    int max_coarsening_level = 30;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> solution;
    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> exact;
    amrex::Array<amrex::MultiFab,AMREX_SPACEDIM> rhs;

    amrex::Real alpha_over_dx2 = 1.e-1;
    amrex::Real alpha;
    amrex::Real beta = 1.0;
};

#endif
