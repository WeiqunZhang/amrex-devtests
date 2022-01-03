#include <AMReX_BiCGSTAB.H>
#include <AMReX_SpMV.H>

namespace amrex {

void bicgstab_solve (AlgVector<Real>& x, SpMatrix<Real> const& A, AlgVector<Real> const& b,
                     Real eps_rel, Real eps_abs)
{
    AlgVector<Real> r(x.partition());

    // compute residual: r = b - A*x;

    SpMV(r, A, x);

    amrex::ForEach(r, b,
    [=] AMREX_GPU_DEVICE (Real& ri, Real const& bi) noexcept
    {
        ri = bi - ri;
    });

    r.printToFile("Ax");
}

}
