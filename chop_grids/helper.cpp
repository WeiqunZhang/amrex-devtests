#include "my_funcs.H"

#if 0
        {
            Long vmin = std::numeric_limits<Long>::max();
            Long vmax = -1;
            int lmax = -1;
            int smin = std::numeric_limits<int>::max();

            int imax = std::numeric_limits<int>::lowest();
            int imin = std::numeric_limits<int>::lowest();
#ifdef AMREX_USE_OMP
#pragma omp parallel
#endif
            {
                Long vmin_this = std::numeric_limits<Long>::max();
                Long vmax_this = -1;
                int lmax_this = -1;
                int smin_this = std::numeric_limits<int>::max();
                int imax_this = std::numeric_limits<int>::lowest();
                int imin_this = std::numeric_limits<int>::lowest();
#ifdef AMREX_USE_OMP
#pragma omp for
#endif
                for (int k = 0; k < bs.size(); k++) {
                    const Box& bx = bs[k];
                    Long v = bx.volume();
                    int ss = bx.shortside();
                    int ls = bx.longside();
                    if (v < vmin_this || (v == vmin_this && ss < smin_this)) {
                        vmin_this = v;
                        smin_this = ss;
                        imin_this = k;
                    }
                    if (v > vmax_this || (v == vmax_this && ls > lmax_this)) {
                        vmax_this = v;
                        lmax_this = ls;
                        imax_this = k;
                    }
                }
#ifdef AMREX_USE_OMP
#pragma omp critical (amr_prtgs)
#endif
                {
                    if (vmin_this < vmin || (vmin_this == vmin && smin_this < smin)) {
                        vmin = vmin_this; // NOLINT
                        smin = smin_this; // NOLINT
                        imin = imin_this;
                    }
                    if (vmax_this > vmax || (vmax_this == vmax && lmax_this > lmax)) {
                        vmax = vmax_this; // NOLINT
                        lmax = lmax_this; // NOLINT
                        imax = imax_this;
                    }
                }
            }
            const Box& bmin = bs[imin];
            const Box& bmax = bs[imax];
            amrex::Print() << "           "
               << " smallest grid: "
                AMREX_D_TERM(<< bmin.length(0),
                             << " x " << bmin.length(1),
                             << " x " << bmin.length(2))
               << "  biggest grid: "
                AMREX_D_TERM(<< bmax.length(0),
                             << " x " << bmax.length(1),
                             << " x " << bmax.length(2))
               << '\n';
        }
#endif
