#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

class MyTest
{
public:

    MyTest ();

    void solve ();
    void writePlotfile () const;

public: // make these public for cuda
    void initProbABecLaplacian ();

private:

    void readParameters ();
    void initData ();
    void solveABecLaplacian ();

    int robin_dir = 0;
    int robin_face = 0;

    int n_cell = 128;
    int max_grid_size = 64;

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 0;
    int max_iter = 100;
    int max_fmg_iter = 0;
    int max_coarsening_level = 30;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab rhs;
    amrex::MultiFab exact_solution;
    amrex::MultiFab acoef;
    amrex::MultiFab bcoef;
    amrex::MultiFab robin_a;
    amrex::MultiFab robin_b;
    amrex::MultiFab robin_f;

    amrex::Real ascalar = 2.0;
    amrex::Real bscalar = 3.0;
};

#endif
