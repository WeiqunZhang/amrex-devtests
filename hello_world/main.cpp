#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Algorithm.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        amrex::Print() << std::numeric_limits<float>::epsilon() * 1.0 * 5 << std::endl;
        amrex::Print() << std::numeric_limits<double>::epsilon() * 1.0 * 5 << std::endl;
        amrex::Print() << "Hello world from AMReX " << amrex::Version() << std::endl;
    }
    amrex::Finalize();
}
