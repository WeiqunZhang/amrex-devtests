#include <AMReX.H>
#include <AMReX_Print.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << "Hello world from AMReX " << amrex::Version << std::endl;
    }
    amrex::Finalize();
}
