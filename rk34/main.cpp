
#include <new>
#include <iostream>
#include <iomanip>

#include <AMReX_Amr.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_AmrLevel.H>

using namespace amrex;

amrex::LevelBld* getLevelBld ();

int
main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    int  max_step = 100000000;
    Real strt_time = 0.;
    Real stop_time = 2.;

    {
        Amr amr(getLevelBld());

        amr.init(strt_time,stop_time);

        while ( amr.okToContinue() &&
                 (amr.levelSteps(0) < max_step || max_step < 0) &&
               (amr.cumTime() < stop_time || stop_time < 0.0) )

        {
            amr.coarseTimeStep(stop_time);
        }
    }

    amrex::Finalize();

    return 0;
}
