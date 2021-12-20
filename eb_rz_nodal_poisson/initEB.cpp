
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include <AMReX_ParmParse.H>

#include <cmath>
#include <algorithm>

#include "MyTest.H"

using namespace amrex;

void
MyTest::initializeEB ()
{
    EB2::Build(geom.back(), max_level, max_level+20);
}
