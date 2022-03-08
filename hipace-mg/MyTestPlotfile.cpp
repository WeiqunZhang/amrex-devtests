
#include "MyTest.H"
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void
MyTest::writePlotfile () const
{
#if 0
    Vector<std::string> varname = {"solution", "rhs", "exact_solution", "error", "acoef"};
    int ncomp = varname.size();

    MultiFab plotmf(grids, dmap, ncomp, 0);
    MultiFab::Copy(plotmf, solution      , 0, 0, 1, 0);
    MultiFab::Copy(plotmf, rhs           , 0, 1, 1, 0);
    MultiFab::Copy(plotmf, exact_solution, 0, 2, 1, 0);
    MultiFab::Copy(plotmf, solution      , 0, 3, 1, 0);
    MultiFab::Subtract(plotmf, plotmf, 2, 3, 1, 0); // error = soln - exact
    MultiFab::Copy(plotmf, acoef, 0, 4, 1, 0);
    amrex::Print() << " max-norm error: " << plotmf.norminf(3) << std::endl;

    WriteSingleLevelPlotfile("plot", plotmf, varname, geom, 0.0, 0);
#endif
}

