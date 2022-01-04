#include <AMReX_BiCGSTAB.H>
#include <AMReX_SpMV.H>

namespace amrex {

void bicgstab_solve (AlgVector<Real>& x, SpMatrix<Real> const& A, AlgVector<Real> const& b,
                     Real eps_rel, Real eps_abs)
{
    AlgVector<Real> xorig(x.partition());
    AlgVector<Real> p    (x.partition());
    AlgVector<Real> r    (x.partition());
    AlgVector<Real> s    (x.partition());
    AlgVector<Real> rh   (x.partition());
    AlgVector<Real> v    (x.partition());
    AlgVector<Real> t    (x.partition());

    SpMV(r, A, x); // r = A*x

    amrex::ForEach(r, b, xorig, x, rh,
    [=] AMREX_GPU_DEVICE (Real& ri, Real const& bi, Real& xoi, Real& xi, Real& rhi) noexcept
    {
        ri = bi - ri;
        xoi = xi;
        xi = 0._rt;
        rhi = ri;
    });

    Real rnorm = r.norminf();
}

}
