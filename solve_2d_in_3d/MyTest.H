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
    void initProbPoisson ();
    void initProbABecLaplacian ();
    void initProbALaplacian ();

private:

    void readParameters ();
    void initData ();
    void solvePoisson ();
    void solveABecLaplacian ();
    void solveALaplacian ();

    int n_cell = 128;
    int max_grid_size = 64;

    int hidden_direction = 1;
    
    int prob_type = 1;  // 1. Poisson,  2. ABecLaplacian,  3. ALaplacian

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

    amrex::Real ascalar = 1.0;
    amrex::Real bscalar = 1.0;
};

#endif
