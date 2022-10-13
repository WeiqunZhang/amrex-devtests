
#include "MyTest.H"
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

void
MyTest::writePlotfile () const
{
    const int nlevels = max_level+1;
    Vector<std::string> varname = {"phi"};
    WriteMultiLevelPlotfile("plot", nlevels, amrex::GetVecOfConstPtrs(phi),
                            varname, geom, 0.0, Vector<int>(nlevels, 0),
                            Vector<IntVect>(nlevels, IntVect{ref_ratio}));
}

