#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        ParmParse::dumpTable(amrex::OutStream(), true);
        amrex::Print() << std::endl;

        ParmParse pp("amrex");
        pp.remove("foo");

        ParmParse::dumpTable(amrex::OutStream(), true);
        amrex::Print() << std::endl;

        ParmParse pp2("");
        pp2.remove("amrex.bar");

        ParmParse::dumpTable(amrex::OutStream(), true);
        amrex::Print() << std::endl;

        pp2.remove("nsteps");
        pp2.remove("stop_time");

        ParmParse::dumpTable(amrex::OutStream(), true);
        amrex::Print() << std::endl;
    }
    amrex::Finalize();
}
