#include "my_funcs.H"

#include <AMReX_Print.H>

void print_inputs (Box const& domain, IntVect const& max_grid_size,
                   IntVect const& blocking_factor, int nprocs)
{
    amrex::Print() << "\n  domain: " << domain << ", max_grid_size: "
                   << max_grid_size << ", blocking_factor: " << blocking_factor
                   << ", nprocs: " << nprocs << "\n";
}

void print_summary (BoxArray const& ba)
{
    Long vmin = std::numeric_limits<Long>::max();
    Long vmax = -1;
    int lmax = -1;
    int smin = std::numeric_limits<int>::max();

    int imax = std::numeric_limits<int>::lowest();
    int imin = std::numeric_limits<int>::lowest();

    for (int k = 0; k < ba.size(); k++) {
        const Box& bx = ba[k];
        Long v = bx.volume();
        int ss = bx.shortside();
        int ls = bx.longside();
        if (v < vmin || (v == vmin && ss < smin)) {
            vmin = v;
            smin = ss;
            imin = k;
        }
        if (v > vmax || (v == vmax && ls > lmax)) {
            vmax = v;
            lmax = ls;
            imax = k;
        }
    }

    const Box& bmin = ba[imin];
    const Box& bmax = ba[imax];
    amrex::Print() << "  " << ba.size() << " grids\n"
                   << "    smallest grid: "
      AMREX_D_TERM(<< bmin.length(0),
                   << " x " << bmin.length(1),
                   << " x " << bmin.length(2))
                   << "  biggest grid: "
      AMREX_D_TERM(<< bmax.length(0),
                   << " x " << bmax.length(1),
                   << " x " << bmax.length(2))
                   << '\n';
}
