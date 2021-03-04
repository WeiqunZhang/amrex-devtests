#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_abeclap (int i, int j, int k,
                          amrex::Array4<amrex::Real> const& rhs,
                          amrex::Array4<amrex::Real> const& sol,
                          amrex::Array4<amrex::Real> const& alpha,
                          amrex::Array4<amrex::Real> const& beta,
                          amrex::Array4<amrex::Real> const& robin_a,
                          amrex::Array4<amrex::Real> const& robin_b,
                          amrex::Array4<amrex::Real> const& robin_f,
                          amrex::Real a, amrex::Real b,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                          const amrex::Dim3& dlo, const amrex::Dim3& dhi,
                          amrex::Box const& vbx, int robin_dir, int robin_face)
{
    constexpr amrex::Real w = 0.05;
    constexpr amrex::Real sigma = 10.;
    const amrex::Real theta = 0.5*std::log(3.) / (w + 1.e-50);

    constexpr amrex::Real pi = 3.1415926535897932;
    constexpr amrex::Real tpi =  2.*pi;
    constexpr amrex::Real fpi =  4.*pi;
    constexpr amrex::Real fac = static_cast<amrex::Real>(AMREX_SPACEDIM*4)*pi*pi;

    amrex::Real xc = (prob_hi[0] + prob_lo[0])*0.5;
    amrex::Real yc = (prob_hi[1] + prob_lo[1])*0.5;
    amrex::Real zc = (prob_hi[2] + prob_lo[2])*0.5;

    amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
    amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
    amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);
    amrex::Real r = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

    beta(i,j,k) = (sigma-1.)/2.*std::tanh(theta*(r-0.25)) + (sigma+1.)/2.;

    x = amrex::min(amrex::max(x,prob_lo[0]),prob_hi[0]);
    y = amrex::min(amrex::max(y,prob_lo[1]),prob_hi[1]);
    z = amrex::min(amrex::max(z,prob_lo[2]),prob_hi[2]);
    r = std::sqrt((x-xc)*(x-xc) + (y-yc)*(y-yc) + (z-zc)*(z-zc));

    amrex::Real tmp = std::cosh(theta*(r-0.25));
    amrex::Real dbdrfac = (sigma-1.)/2./(tmp*tmp) * theta/r;
    dbdrfac *= b;

    sol(i,j,k) = std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
         + .25 * std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z);

    if (vbx.contains(i,j,k)) {
        rhs(i,j,k) = beta(i,j,k)*b*fac*(std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
                                      + std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z))
                 + dbdrfac*((x-xc)*(tpi*std::sin(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
                                   + pi*std::sin(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z))
                          + (y-yc)*(tpi*std::cos(tpi*x) * std::sin(tpi*y) * std::cos(tpi*z)
                                   + pi*std::cos(fpi*x) * std::sin(fpi*y) * std::cos(fpi*z))
                          + (z-zc)*(tpi*std::cos(tpi*x) * std::cos(tpi*y) * std::sin(tpi*z)
                                   + pi*std::cos(fpi*x) * std::cos(fpi*y) * std::sin(fpi*z)))
                                 + a * (std::cos(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
                               + 0.25 * std::cos(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z));
        alpha(i,j,k) = 1.;
    }

    // Robin BC
    bool robin_cell = false;
    if (robin_dir == 0 && j >= dlo.y && j <= dhi.y && k >= dlo.z && k <= dhi.z) {
        robin_cell = (i > dhi.x) || (i < dlo.x);
    } else if (robin_dir == 1 && i >= dlo.x && i <= dhi.x && k >= dlo.z && k <= dhi.z) {
        robin_cell = (j > dhi.y) || (j < dlo.y);
    } else if (robin_dir == 2 && i >= dlo.x && i <= dhi.x && j >= dlo.y && j <= dhi.y) {
        robin_cell = (k > dhi.z) || (k < dlo.z);
    }
    if (robin_cell) {
        if (y+z > 0.0) {
            robin_a(i,j,k) = 3.0;
            robin_b(i,j,k) = 1.0;
        } else {
            robin_a(i,j,k) = 1.0;
            robin_b(i,j,k) = 3.0;
        }
        amrex::Real dphidn = -tpi * std::sin(tpi*x) * std::cos(tpi*y) * std::cos(tpi*z)
                             - pi * std::sin(fpi*x) * std::cos(fpi*y) * std::cos(fpi*z);
        if (robin_face == 0) dphidn *= -1.0;
        robin_f(i,j,k) = robin_a(i,j,k) * sol(i,j,k) + robin_b(i,j,k) * dphidn;
    }
}

#endif
