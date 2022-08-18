#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

using namespace amrex;

void test2 (int i, int j, int k, Array4<Real> const& a, Array4<Real const> const& b, Array4<Real const> const& c)
{
    a(i,j,k) = b(i,j,k) + 3. * c(i,j,k);
}
