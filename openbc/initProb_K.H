#ifndef INIT_PROB_K_H_
#define INIT_PROB_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void actual_init_poisson (int i, int j, int k,
                          amrex::Array4<amrex::Real> const& rhs,
                          amrex::Array4<amrex::Real> const& exact,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& prob_lo,
                          amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
                          amrex::Real r0, amrex::Real dcell)
{
    amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
    amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
    amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);

    amrex::Real r = std::sqrt(x*x+y*y+z*z);

    if (r+0.5*dcell < r0) {
        rhs(i,j,k) = 1.0;
    } else if (r-0.5*dcell > r0) {
        rhs(i,j,k) = 0.0;
    } else {
        int ivol = 0;
        int tvol = 0;
        for         (int kk = 0; kk < 4; ++kk) {
            amrex::Real zz = z + (kk-1.5)*0.25*dx[2];
            for     (int jj = 0; jj < 4; ++jj) {
                amrex::Real yy = y + (jj-1.5)*0.25*dx[1];
                for (int ii = 0; ii < 4; ++ii) {
                    amrex::Real xx = x + (ii-1.5)*0.25*dx[0];
                    amrex::Real rr = std::sqrt(xx*xx+yy*yy+zz*zz);
                    if (rr < r0) {
                        ++ivol;
                    }
                    ++tvol;
                }
            }
        }
        rhs(i,j,k) = static_cast<amrex::Real>(ivol) / static_cast<amrex::Real>(tvol);
    }

    if (r < r0) {
        exact(i,j,k) = -r0*r0/3. + (r*r-r0*r0)/6.;
    } else {
        exact(i,j,k) = -r0*r0*r0/(3.*r);
    }
}

#endif
