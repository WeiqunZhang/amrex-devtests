
#include <AMReX_MultiFab.H>

using namespace amrex;

void initdata (MultiFab& S_tmp, const Geometry& geom)
{
    S_tmp.setVal(1.0);
}
