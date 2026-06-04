#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Reduce.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        std::vector<int> foo(10);
        for (int i = 0; i < 10; ++i) {
            foo[i] = i*10;
        }
        auto mm = amrex::Reduce::MinMax(10, foo.data());
        amrex::Print() << "xxxxx mm: " << mm.first << " " << mm.second << std::endl;
    }
    amrex::Finalize();
}
