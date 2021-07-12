#include <init.H>
#include <work.H>

#include <AMReX.H>

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        init();
        work();
    }
    amrex::Finalize();
}
