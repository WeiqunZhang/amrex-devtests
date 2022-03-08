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

    void initProbALaplacian ();

private:

    void readParameters ();
    void initData ();

    int n_cell = 256;

    // For MLMG solver
    int verbose = 2;
    int max_iter = 100;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab rhs;
    amrex::MultiFab exact_solution;
    amrex::MultiFab acoef;
};

#endif
