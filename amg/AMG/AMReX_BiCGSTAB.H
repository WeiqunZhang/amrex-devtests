#ifndef AMREX_BICGSTAB_H_
#define AMREX_BICGSTAB_H_
#include <AMReX_Config.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>
#include <AMReX_SpMV.H>

namespace amrex {

template <typename T>
int bicgstab_solve (AlgVector<T>& x, SpMatrix<T> const& A, AlgVector<T> const& b,
                    T eps_rel, T eps_abs, int verbose = 0, int maxiter = 200)
{
    AlgVector<T> ph   (x.partition());
    AlgVector<T> sh   (x.partition());
    AlgVector<T> xorig(x.partition());
    AlgVector<T> p    (x.partition());
    AlgVector<T> r    (x.partition());
    AlgVector<T> s    (x.partition());
    AlgVector<T> rh   (x.partition());
    AlgVector<T> v    (x.partition());
    AlgVector<T> t    (x.partition());

    SpMV(r, A, x); // r = A*x

    amrex::ForEach(r, b, xorig, x, rh,
    [=] AMREX_GPU_DEVICE (T& ri, T const& bi, T& xoi, T& xi, T& rhi) noexcept
    {
        ri = bi - ri;
        xoi = xi;
        xi = 0;
        rhi = ri;
    });

    T rnorm = r.norminf();
    const T rnorm0 = rnorm;

    if (verbose > 0) {
        amrex::Print() << "bicgstab_solve: Initial error (error0) = " << rnorm0 << '\n';
    }

    int ret = 0;
    int iter = 1;
    T rho_1 = 0, alpha = 0, omega = 0;

    if (rnorm0 == 0 || rnorm0 < eps_abs) {
        if (verbose > 0) {
            amrex::Print() << "bicgstab_solve: niter = 0,"
                           << ", rnorm = " << rnorm
                           << ", eps_abs = " << eps_abs << '\n';
        }
        return ret;
    }

    for (; iter <= maxiter; ++iter) {
        const T rho = amrex::Dot(rh,r);
        if (rho == 0) {
            ret = 1;
            break;
        }

        if (iter == 1) {
            p.copyAsync(r);
        } else {
            const T beta = (rho/rho_1)*(alpha/omega);
            amrex::ForEach(p, r, v,
            [=] AMREX_GPU_DEVICE (T& pi, T const& ri, T const& vi) noexcept
            {
                pi = ri + beta * (pi - omega * vi);
            });
        }

        ph.copyAsync(p);

        SpMV(v, A, ph);

        T rhTv = amrex::Dot(rh,v);
        if (rhTv != 0) {
            alpha = rho/rhTv;
        } else {
            ret = 2;
            break;
        }

        amrex::ForEach(x, ph, s, r, v,
        [=] AMREX_GPU_DEVICE (T& xi, T const& phi, T& si, T const& ri, T const& vi) noexcept
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

        T tvals[2];
        {
            ReduceOps<ReduceOpSum,ReduceOpSum> reduce_op;
            ReduceData<T,T> reduce_data(reduce_op);
            Long n = x.numLocalRows();
            T const* pt = t.data();
            T const* ps = s.data();
            reduce_op.eval(n, reduce_data, [=] AMREX_GPU_DEVICE (Long i) noexcept
                           {
                               return makeTuple(pt[i]*pt[i], pt[i]*ps[i]);
                           });
            auto hv = reduce_data.value(reduce_op);
            tvals[0] = amrex::get<0>(hv);
            tvals[1] = amrex::get<1>(hv);
        }
        ParallelAllReduce::Sum(tvals, 2, ParallelContext::CommunicatorSub());

        if (tvals[0] != 0) {
            omega = tvals[1] / tvals[0];
        } else {
            ret = 3;
            break;
        }

        amrex::ForEach(x, sh, r, s, t,
        [=] AMREX_GPU_DEVICE (T& xi, T const& shi, T& ri, T const& si, T const& ti) noexcept
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

        if (omega == 0) {
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

#endif
