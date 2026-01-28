#include <AMReX.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Print.H>
#include "my_funcs.H"

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    // WarpX case
    // https://github.com/AMReX-Codes/amrex/issues/4746
#if (AMREX_SPACEDIM == 3)
    {
        IntVect n_cell{576, 576, 432};
        Box domain(IntVect(0), n_cell-1);
        IntVect max_grid_size(48,48,72);
        IntVect blocking_factor(8);
        int nprocs = 864; // 256;
        auto ba = v4::make_base_grids(domain,max_grid_size,blocking_factor,nprocs);
        print_summary(ba);
    }
#endif

    amrex::Finalize();
}
