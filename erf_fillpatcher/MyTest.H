#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MultiFab.H>

class MyTest
{
public:

    MyTest ();

    void test ();
    void writePlotfile () const;

    void initData ();

private:

    void readParameters ();

    int max_level = 1;
    int ref_ratio = 2;
    int n_cell = 128;
    int max_grid_size = 32;

    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::BoxArray> grids;
    amrex::Vector<amrex::DistributionMapping> dmap;

    amrex::Vector<amrex::MultiFab> phi;
};

#endif
