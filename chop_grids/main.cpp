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
        Box domain(IntVect(0), IntVect(575,575,431));
        BoxArray bs = amrex::decompose(domain, 864);
        amrex::Print() << "xxxxx bs.size() = " << bs.size() << std::endl;

    }
#endif

    amrex::Finalize();
}
