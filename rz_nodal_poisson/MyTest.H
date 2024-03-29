#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>
#include <AMReX_Array.H>
#include <string>

class MyTest
{
public:

    MyTest ();

    void solve ();

    void writePlotfile ();

    void initData ();

private:

    void readParameters ();
    void initGrids ();
    void addFineGrids ();

    int max_level = 0;
    int ref_ratio = 2;
    int n_cell = 128;
    int max_grid_size = 64;

    std::string plot_file_name{"plot"};

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 2;
    int max_iter = 100;
    int max_fmg_iter = 0;
    int max_bottom_iter = 1000;
    amrex::Real reltol        = 1.e-12;
    amrex::Real bottom_reltol = 1.e-4;
    int max_coarsening_level = 30;
    bool use_hypre = false;
    bool use_petsc = false;
    int agg_grid_size = -1;
    int con_grid_size = -1;

    int composite_solve = 0;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    amrex::Vector<amrex::MultiFab> exact;
    amrex::Vector<amrex::MultiFab> phi;
    amrex::Vector<amrex::MultiFab> rhs;
};

#endif
