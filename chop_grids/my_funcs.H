#ifndef MY_FUNCS_H_
#define MY_FUNCS_H_

#include <AMReX_BoxArray.H>

using namespace amrex; // NOLINT

void print_inputs (Box const& domain, IntVect const& max_grid_size,
                   IntVect const& blocking_factor, int nprocs);

void print_summary (BoxArray const& ba);

namespace v0 {
    BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                              IntVect const& blocking_factor, int nprocs);
}

namespace v1 {
    BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                              IntVect const& blocking_factor, int nprocs);
}

namespace v2 {
    BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                              IntVect const& blocking_factor, int nprocs);
}

namespace v3 {
    BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                              IntVect const& blocking_factor, int nprocs);
}

namespace v4 {
    BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                              IntVect const& blocking_factor, int nprocs);
}

#endif
