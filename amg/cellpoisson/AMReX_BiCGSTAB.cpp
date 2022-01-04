#include <AMReX_BiCGSTAB.H>
#include <AMReX_SpMV.H>

namespace amrex {

int bicgstab_solve (AlgVector<Real>& x, SpMatrix<Real> const& A, AlgVector<Real> const& b,
                    Real eps_rel, Real eps_abs, int verbose)
{
    constexpr int maxiter = 200;

    AlgVector<Real> ph   (x.partition());
    AlgVector<Real> sh   (x.partition());
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
    const Real rnorm0 = rnorm;

    if (verbose > 0) {
        amrex::Print() << "bicgstab_solve: Initial error (error0) = " << rnorm0 << '\n';
    }

    int ret = 0;
    int iter = 1;
    Real rho_1 = 0, alpha = 0, omega = 0;

    if (rnorm0 == 0 || rnorm0 < eps_abs) {
        if (verbose > 0) {
            amrex::Print() << "bicgstab_solve: niter = 0,"
                           << ", rnorm = " << rnorm
                           << ", eps_abs = " << eps_abs << '\n';
        }
        return ret;
    }

    for (; iter <= maxiter; ++iter) {
        const Real rho = amrex::Dot(rh,r);
        if (rho == 0._rt) {
            ret = 1;
            break;
        }

        if (iter == 1) {
            p.copyAsync(r);
        } else {
            const Real beta = (rho/rho_1)*(alpha/omega);
            amrex::ForEach(p, r, v,
            [=] AMREX_GPU_DEVICE (Real& pi, Real const& ri, Real const& vi) noexcept
            {
                pi = ri + beta * (pi - omega * vi);
            });
        }

        ph.copyAsync(p);

        SpMV(v, A, ph);

        Real rhTv = amrex::Dot(rh,v);
        if (rhTv != 0._rt) {
            alpha = rho/rhTv;
        } else {
            ret = 2; break;
        }

        amrex::ForEach(x, ph, s, r, v,
        [=] AMREX_GPU_DEVICE (Real& xi, Real const& phi, Real& si, Real const& ri,
                              Real const& vi) noexcept
        {
            xi += alpha * phi;
            si = ri - alpha * vi;
        });

        rnorm = s.norminf();

        if (verbose > 2) {
            amrex::Print() << "bicgstab_solve: Half Iter "
                           << std::setw(11) << iter
                           << " rel. err. "
                           << rnorm/(rnorm0) << '\n';
        }

        if (rnorm < eps_rel * rnorm0 || rnorm < eps_abs) { break; }

        sh.copyAsync(s);

        SpMV(t, A, sh);

        Real tvals[2];
        {
            ReduceOps<ReduceOpSum,ReduceOpSum> reduce_op;
            ReduceData<Real,Real> reduce_data(reduce_op);
            Long n = x.numLocalRows();
            Real const* pt = t.data();
            Real const* ps = s.data();
            reduce_op.eval(n, reduce_data, [=] AMREX_GPU_DEVICE (Long i) noexcept
                           {
                               return makeTuple(pt[i]*pt[i], pt[i]*ps[i]);
                           });
            auto hv = reduce_data.value(reduce_op);
            tvals[0] = amrex::get<0>(hv);
            tvals[1] = amrex::get<1>(hv);
        }
        ParallelAllReduce::Sum(tvals, 2, ParallelContext::CommunicatorSub());

        if (tvals[0] != 0._rt) {
            omega = tvals[1] / tvals[0];
        } else {
            ret = 3;
            break;
        }

        amrex::ForEach(x, sh, r, s, t,
        [=] AMREX_GPU_DEVICE (Real& xi, Real const& shi, Real& ri, Real const& si,
                              Real const& ti) noexcept
        {
            xi += omega * shi;
            ri = si - omega * ti;
        });

        rnorm = r.norminf();

        if (verbose > 2) {
            amrex::Print() << "bicgstab_solve: Iteration "
                           << std::setw(11) << iter
                           << " rel. err. "
                           << rnorm/(rnorm0) << '\n';
        }

        if (rnorm < eps_rel*rnorm0 || rnorm < eps_abs) { break; }

        if (omega == 0._rt) {
            ret = 4;
            break;
        }
        rho_1 = rho;
    }

    if (verbose > 0) {
        amrex::Print() << "bicgstab_solve: Final: Iteration "
                       << std::setw(4) << iter
                       << " rel. err. "
                       << rnorm/(rnorm0) << '\n';
    }

    if (ret == 0 && rnorm > eps_rel*rnorm0 && rnorm > eps_abs) {
        if ( verbose > 0) {
            amrex::Warning("bicgstab_solve: failed to converge!");
        }
        ret = 8;
    }

    if ((ret == 0 || ret == 8) && (rnorm < rnorm0)) {
        x.plus(xorig);
    } else {
        x.copy(xorig);
    }

    return ret;
}

}
