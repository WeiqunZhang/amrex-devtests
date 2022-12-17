
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include "MyTest.H"

int main (int argc, char* argv[])
{
    {
        amrex::ParmParse pp("amrex");
        pp.add("the_arena_is_managed",0);
    }

    amrex::Initialize(argc, argv);

    {
        MyTest mytest;
        mytest.solve();
        auto t0 = amrex::second();
        mytest.solve();
        auto t1 = amrex::second();
        amrex::Print() << "\nSolve Time is " << t1-t0 << "\n\n";
    }

    amrex::Finalize();
}
