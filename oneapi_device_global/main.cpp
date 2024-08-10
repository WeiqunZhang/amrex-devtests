#include <AMReX.H>

void init();
void work();

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        init();
        work();
    }
    amrex::Finalize();
}
