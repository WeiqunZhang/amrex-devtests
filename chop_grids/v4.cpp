// amrex b7083f0b20d Wed Aug 14 00:16:24 2024

#include "my_funcs.H"

using namespace amrex;

namespace v4 {

static void ChopGrids (BoxArray& ba, int target_size, IntVect const& max_grid_size,
                       IntVect const& blocking_factor, IntVect const& ref_ratio,
                       Box const& domain)
{
    IntVect chunk = max_grid_size;
    chunk.min(domain.length());

    // Note that ba already satisfies the max_grid_size requirement and it's
    // coarsenable if it's a fine level BoxArray.

    while (ba.size() < target_size)
    {
        IntVect chunk_prev = chunk;

        std::array<std::pair<int,int>,AMREX_SPACEDIM>
            chunk_dir{AMREX_D_DECL(std::make_pair(chunk[0],int(0)),
                                   std::make_pair(chunk[1],int(1)),
                                   std::make_pair(chunk[2],int(2)))};
        std::sort(chunk_dir.begin(), chunk_dir.end());

        for (int idx = AMREX_SPACEDIM-1; idx >= 0; idx--) {
            int idim = chunk_dir[idx].second;
            int new_chunk_size = chunk[idim] / 2;
            int rr = ref_ratio[idim];
            if (rr > 1) {
                new_chunk_size = (new_chunk_size/rr) * rr;
            }
            if (new_chunk_size != 0 &&
                new_chunk_size%blocking_factor[idim] == 0)
            {
                chunk[idim] = new_chunk_size;
                if (rr == 1) {
                    ba.maxSize(chunk);
                } else {
                    IntVect bf(1);
                    bf[idim] = rr;
                    // Note that only idim-direction will be chopped by
                    // minmaxSize because the sizes in other directions
                    // are already smaller than chunk.
                    ba.minmaxSize(bf, chunk);
                }
                break;
            }
        }

        if (chunk == chunk_prev) {
            break;
        }
    }
}

BoxArray make_base_grids (Box const& domain, IntVect const& max_grid_size,
                          IntVect const& blocking_factor, int nprocs)
{
    print_inputs(domain, max_grid_size, blocking_factor, nprocs);

    IntVect fac(2);
    const Box dom2 = amrex::refine(amrex::coarsen(domain,2),2);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (domain.length(idim) != dom2.length(idim)) {
            fac[idim] = 1;
        }
    }
    BoxArray ba(amrex::coarsen(domain,fac));
    ba.maxSize(max_grid_size/fac);

    ChopGrids(ba, nprocs, max_grid_size, blocking_factor, IntVect(1), domain);

    return ba;
}

}
