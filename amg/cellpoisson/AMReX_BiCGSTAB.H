#ifndef AMREX_BICGSTAB_H_
#define AMREX_BICGSTAB_H_
#include <AMReX_Config.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

namespace amrex {

int bicgstab_solve (AlgVector<Real>& x, SpMatrix<Real> const& A, AlgVector<Real> const& b,
                    Real eps_rel, Real eps_abs, int verbose = 0);

}

#endif
