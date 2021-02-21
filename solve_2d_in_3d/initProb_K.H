#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_poisson (int i, int j, int k,
                          amrex::Array4<amrex::Real> const& rhs,
                          amrex::Array4<amrex::Real> const& exact,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                          int hd)
{
    constexpr amrex::Real tpi = 2.*3.1415926535897932;
    constexpr amrex::Real fpi = 4.*3.1415926535897932;
    constexpr amrex::Real fac = tpi*tpi*2;
    amrex::Real x = amrex::min(amrex::max(prob_lo[0] + dx[0] * (i + 0.5), prob_lo[0]), prob_hi[0]);
    amrex::Real y = amrex::min(amrex::max(prob_lo[1] + dx[1] * (j + 0.5), prob_lo[1]), prob_hi[1]);
    amrex::Real z = amrex::min(amrex::max(prob_lo[2] + dx[2] * (k + 0.5), prob_lo[2]), prob_hi[2]);

    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> hdf2{std::sin(tpi*x),
                                                     std::sin(tpi*y),
                                                     std::sin(tpi*z)};
    hdf2[hd] = 1.0;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> hdf4{std::sin(fpi*x),
                                                     std::sin(fpi*y),
                                                     std::sin(fpi*z)};
    hdf4[hd] = 1.0;

    exact(i,j,k) = hdf2[0]*hdf2[1]*hdf2[2] + .25*hdf4[0]*hdf4[1]*hdf4[2];
    if (rhs.contains(i,j,k)) {
        rhs(i,j,k) = -fac*(hdf2[0]*hdf2[1]*hdf2[2] + hdf4[0]*hdf4[1]*hdf4[2]);
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_alaplacian (int i, int j, int k,
                             amrex::Array4<amrex::Real> const& rhs,
                             amrex::Array4<amrex::Real> const& exact,
                             amrex::Array4<amrex::Real> const& acoef,
                             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_hi,
                             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                             int hd)
{
    constexpr amrex::Real tpi = 2.*3.1415926535897932;
    constexpr amrex::Real fpi = 4.*3.1415926535897932;
    constexpr amrex::Real fac = tpi*tpi*2;
    amrex::Real x = amrex::min(amrex::max(prob_lo[0] + dx[0] * (i + 0.5), prob_lo[0]), prob_hi[0]);
    amrex::Real y = amrex::min(amrex::max(prob_lo[1] + dx[1] * (j + 0.5), prob_lo[1]), prob_hi[1]);
    amrex::Real z = amrex::min(amrex::max(prob_lo[2] + dx[2] * (k + 0.5), prob_lo[2]), prob_hi[2]);

    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> hdf2{std::sin(tpi*x),
                                                     std::sin(tpi*y),
                                                     std::sin(tpi*z)};
    hdf2[hd] = 1.0;
    amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> hdf4{std::sin(fpi*x),
                                                     std::sin(fpi*y),
                                                     std::sin(fpi*z)};
    hdf4[hd] = 1.0;

    exact(i,j,k) = hdf2[0]*hdf2[1]*hdf2[2] + .25*hdf4[0]*hdf4[1]*hdf4[2];
    if (rhs.contains(i,j,k)) {
        acoef(i,j,k) = 1.0 + 0.5*hdf2[0]*hdf4[1]*hdf4[2];
        rhs(i,j,k) = acoef(i,j,k) * exact(i,j,k)
            + fac*(hdf2[0]*hdf2[1]*hdf2[2] + hdf4[0]*hdf4[1]*hdf4[2]);
    }
}

#endif
