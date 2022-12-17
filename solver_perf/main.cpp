
#include <AMReX.H>
#include <AMReX_Print.H>
#include "MyTest.H"

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        MyTest mytest;
        mytest.solve();
        auto t0 = amrex::second();
        mytest.solve();
        auto t1 = amrex::second();
        amrex::Print() << "Solve Time is " << t1-t0 << "\n";
    }

    amrex::Finalize();
}
