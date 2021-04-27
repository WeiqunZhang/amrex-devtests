#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

using namespace amrex;

void main_main ()
{
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
