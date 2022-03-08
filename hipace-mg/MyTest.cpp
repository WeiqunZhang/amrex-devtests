#include "HpMultiGrid.H"

#include "MyTest.H"

#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLALaplacian.H>
#include <AMReX_MLPoisson.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::solve ()
{
#if 0

    LPInfo lpinfo;
    lpinfo.setMaxCoarseningLevel(30).setHiddenDirection(2);
    MLALaplacian mlalaplacian({geom}, {grids}, {dmap}, lpinfo);

    // This is a 3d problem with Dirichlet BC
    mlalaplacian.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet)},
                            {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet)});

    mlalaplacian.setLevelBC(0, &solution);

    mlalaplacian.setScalars(-1.0, -1.0);

    mlalaplacian.setACoeffs(0, acoef);

    MLMG mlmg(mlalaplacian);
    mlmg.setVerbose(verbose);

    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);

#else

    AMREX_ALWAYS_ASSERT(solution.size() == 1);
    hpmg::MultiGrid mg(geom.Domain());
    mg.solve(solution[0], rhs[0], acoef[0], geom.CellSize(0), geom.CellSize(1),
             1.e-10, 0.0, 20, verbose);

#endif
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("n_cell", n_cell);
    pp.query("verbose", verbose);
}

void
MyTest::initData ()
{
    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    domain0.setRange(2, n_cell/2);
    geom.define(domain0);

    grids.define(domain0);

    IntVect ng = IntVect{1};
    ng[2] = 0;

    dmap.define(grids);
    solution      .define(grids, dmap, 2, ng);
    rhs           .define(grids, dmap, 2, 0);
    exact_solution.define(grids, dmap, 2, ng);
    acoef.define(grids, dmap, 1, 0);

    initProbALaplacian();
}

