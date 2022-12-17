#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_poisson (int i, int j, int k,
                          amrex::Array4<amrex::Real> const& rhs,
                          amrex::Array4<amrex::Real> const& sol,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    constexpr amrex::Real tpi = 2.*3.1415926535897932;
    constexpr amrex::Real fpi = 4.*3.1415926535897932;
    constexpr amrex::Real fac = tpi*tpi*static_cast<amrex::Real>(AMREX_SPACEDIM);
    amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
    amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
    x = amrex::min(amrex::max(x,prob_lo[0]),prob_hi[0]);
    y = amrex::min(amrex::max(y,prob_lo[1]),prob_hi[1]);

#if (AMREX_SPACEDIM == 2)

    sol(i,j,k) = (std::sin(tpi*x) * std::sin(tpi*y))
         + .25 * (std::sin(fpi*x) * std::sin(fpi*y));

    if (rhs.contains(i,j,k)) {
        rhs(i,j,k) = -fac * (std::sin(tpi*x) * std::sin(tpi*y))
                     -fac * (std::sin(fpi*x) * std::sin(fpi*y));
    }

#else

    amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);
    z = amrex::min(amrex::max(z,prob_lo[2]),prob_hi[2]);

    sol(i,j,k) = (std::sin(tpi*x) * std::sin(tpi*y) * std::sin(tpi*z))
         + .25 * (std::sin(fpi*x) * std::sin(fpi*y) * std::sin(fpi*z));

    if (rhs.contains(i,j,k)) {
        rhs(i,j,k) = -fac * (std::sin(tpi*x) * std::sin(tpi*y) * std::sin(tpi*z))
                     -fac * (std::sin(fpi*x) * std::sin(fpi*y) * std::sin(fpi*z));
    }

#endif
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_abeclap (int i, int j, int k,
                          amrex::Array4<amrex::Real> const& rhs,
                          amrex::Array4<amrex::Real> const& sol,
                          amrex::Real a, amrex::Real b,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx)
{
    constexpr amrex::Real pi = 3.1415926535897932;
    constexpr amrex::Real tpi =  2.*pi;
    constexpr amrex::Real fpi =  4.*pi;
    constexpr amrex::Real fac = static_cast<amrex::Real>(AMREX_SPACEDIM*4)*pi*pi;

    amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
    x = amrex::min(amrex::max(x,prob_lo[0]),prob_hi[0]);

    amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
    y = amrex::min(amrex::max(y,prob_lo[1]),prob_hi[1]);

#if (AMREX_SPACEDIM == 2)
    amrex::Real z = 0.0;
#else
    amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);
    z = amrex::min(amrex::max(z,prob_lo[2]),prob_hi[2]);
#endif

    sol(i,j,k) = std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
         + .25 * std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z);

    if (rhs.contains(i,j,k)) {
        rhs(i,j,k) = a*sol(i,j,k)
            + b*fac*(std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
                   + std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z));
    }
}
