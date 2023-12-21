#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_prob (int i, int j, int k,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,AMREX_SPACEDIM> const& rhs,
                       amrex::GpuArray<amrex::Array4<amrex::Real>,AMREX_SPACEDIM> const& sol,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& problo,
                       amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                       amrex::Real alpha, amrex::Real beta)
{
    using namespace amrex;

    constexpr amrex::Real tpi = Real(2.)*amrex::Math::pi<Real>();
    constexpr amrex::Real fpi = Real(4.)*amrex::Math::pi<Real>();

    Real xnd = problo[0] + Real(i)*dx[0];
    Real ynd = problo[1] + Real(j)*dx[1];
    Real znd = problo[2] + Real(k)*dx[2];
    Real xcc = xnd + Real(0.5)*dx[0];
    Real ycc = ynd + Real(0.5)*dx[1];
    Real zcc = znd + Real(0.5)*dx[2];

    if (sol[0].contains(i,j,k)) {
        Real x = xcc;
        Real y = ynd;
        Real z = znd;
        sol[0](i,j,k) = std::cos(tpi*x) * std::sin(fpi*y) * std::sin(fpi*z);
     }

    if (sol[1].contains(i,j,k)) {
        Real x = xnd;
        Real y = ycc;
        Real z = znd;
        sol[1](i,j,k) = std::sin(fpi*x) * std::cos(tpi*y) * std::sin(fpi*z);
    }

    if (sol[2].contains(i,j,k)) {
        Real x = xnd;
        Real y = ynd;
        Real z = zcc;
        sol[2](i,j,k) = std::sin(fpi*x) * std::sin(fpi*y) * std::cos(tpi*z);
    }

    if (rhs[0].contains(i,j,k)) {
        Real x = xcc;
        Real y = ynd;
        Real z = znd;
        rhs[0](i,j,k) = (beta + alpha*Real(4.0)*tpi*fpi) * sol[0](i,j,k)
            - alpha*tpi*fpi * std::cos(fpi*x) * (std::sin(tpi*y) * std::sin(fpi*z) +
                                                 std::sin(fpi*y) * std::sin(tpi*z));
    }

    if (rhs[1].contains(i,j,k)) {
        Real x = xnd;
        Real y = ycc;
        Real z = znd;
        rhs[1](i,j,k) = (beta + alpha*Real(4.0)*tpi*fpi) * sol[1](i,j,k)
            - alpha*tpi*fpi * std::cos(fpi*y) * (std::sin(tpi*x) * std::sin(fpi*z) +
                                                 std::sin(fpi*x) * std::sin(tpi*z));
    }

    if (rhs[2].contains(i,j,k)) {
        Real x = xnd;
        Real y = ynd;
        Real z = zcc;
        rhs[2](i,j,k) = (beta + alpha*Real(4.0)*tpi*fpi) * sol[2](i,j,k)
            - alpha*tpi*fpi * std::cos(fpi*z) * (std::sin(tpi*x) * std::sin(fpi*y) +
                                                 std::sin(fpi*x) * std::sin(tpi*y));
    }
}

#endif
