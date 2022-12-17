#include "MyTest.H"

#include <AMReX_ParmParse.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::solve ()
{
    if (prob_type == 1) {
        if (single_precision) {
            solvePoisson<fMultiFab>();
        } else {
            solvePoisson<MultiFab>();
        }
    } else if (prob_type == 2) {
        if (single_precision) {
            solveABecLaplacian<fMultiFab>();
        } else {
            solveABecLaplacian<MultiFab>();
        }
    } else {
        amrex::Abort("Unknown prob_type");
    }
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.queryarr("n_cells", n_cells);
    pp.query("max_grid_size", max_grid_size);

    pp.query("prob_type", prob_type);

    pp.query("single_precision", single_precision);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("linop_maxorder", linop_maxorder);
    pp.query("agglomeration", agglomeration);
    pp.query("consolidation", consolidation);
    pp.query("max_coarsening_level", max_coarsening_level);
}

void
MyTest::initData ()
{
    const Real dx = 1./512.;
    std::array<Real,AMREX_SPACEDIM> xmax{AMREX_D_DECL(dx*n_cells[0],
                                                      dx*n_cells[1],
                                                      dx*n_cells[2])};

    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, xmax);
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain(IntVect(0), IntVect{AMREX_D_DECL(n_cells[0]-1,
                                                n_cells[1]-1,
                                                n_cells[2]-1)});
    geom.define(domain);
    grids.define(domain);
    grids.maxSize(max_grid_size);
    dmap.define(grids);
    solution.define(grids, dmap, 1, 1);
    rhs     .define(grids, dmap, 1, 0);
    if (prob_type == 2) {
        acoef.define(grids, dmap, 1, 0);
        bcoef.define(grids, dmap, 1, 1);
    }

    if (prob_type == 1) {
        initProbPoisson();
    } else if (prob_type == 2) {
        initProbABecLaplacian();
    } else {
        amrex::Abort("Unknown prob_type "+std::to_string(prob_type));
    }
}
