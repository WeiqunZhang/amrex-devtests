
#include "MyTest.H"
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

void
MyTest::writePlotfile () const
{
    Vector<std::string> varname = {"solution", "rhs", "exact_solution", "error"};
    int ncomp = varname.size();
    if (!acoef.empty()) {
        varname.emplace_back("acoef");
        ++ncomp;
    }
    if (!bcoef.empty()) {
        varname.emplace_back("bcoef");
        ++ncomp;
    }

    MultiFab plotmf(grids, dmap, ncomp, 0);
    MultiFab::Copy(plotmf, solution      , 0, 0, 1, 0);
    MultiFab::Copy(plotmf, rhs           , 0, 1, 1, 0);
    MultiFab::Copy(plotmf, exact_solution, 0, 2, 1, 0);
    MultiFab::Copy(plotmf, solution      , 0, 3, 1, 0);
    MultiFab::Subtract(plotmf, plotmf, 2, 3, 1, 0); // error = soln - exact
    if (!acoef.empty()) {
        MultiFab::Copy(plotmf, acoef, 0, 4, 1, 0);
    }
    if (!bcoef.empty()) {
        MultiFab::Copy(plotmf, bcoef, 0, 5, 1, 0);
    }
    amrex::Print() << " max-norm error: " << plotmf.norminf(3) << std::endl;

    WriteSingleLevelPlotfile("plot", plotmf, varname, geom, 0.0, 0);
}

