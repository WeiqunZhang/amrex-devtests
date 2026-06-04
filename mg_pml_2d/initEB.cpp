
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include "MyTest.H"

using namespace amrex;

void
MyTest::initializeEB ()
{
    EB2::SphereIF sphere0(prob_a, RealArray{-0.5*prob_d,0.0}, false);
    EB2::SphereIF sphere1(prob_a, RealArray{+0.5*prob_d,0.0}, false);
    auto two_spheres = EB2::makeUnion(sphere0, sphere1);
    auto gshop = EB2::makeShop(two_spheres);
    EB2::Build(gshop, geom[0], 0, 20);
}
