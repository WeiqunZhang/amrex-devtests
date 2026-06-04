#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_EBFabFactory.H>
#include <AMReX_MLMG.H>
#include <AMReX_Array.H>
#include <string>

class MyTest
{
public:

    MyTest ();

    void solve ();

    void solve0 ();
    void solve1 ();
    void solve2 ();

    void writePlotfile ();

    void initData ();

private:

    void initializeEB ();
    void readParameters ();
    void initGrids ();

    int method = 2;
    int n_cell_pml = 16;


    amrex::Real prob_a = 0.12;
    amrex::Real prob_d = 0.45;

    int n_cell = 128;
    int max_grid_size = 64;

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 2;
    int max_iter = 100;
    int max_bottom_iter = 100000;
    amrex::Real reltol        = 1.e-11;
    amrex::Real bottom_reltol = 1.e-4;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;
    amrex::Vector<std::unique_ptr<amrex::EBFArrayBoxFactory> > factory;

    amrex::Vector<amrex::MultiFab> phi;
    amrex::Vector<amrex::MultiFab> exact;
    amrex::Vector<amrex::MultiFab> rhs;
};

#endif
