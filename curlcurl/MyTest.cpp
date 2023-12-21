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
    LPInfo info;
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setMaxCoarseningLevel(max_coarsening_level);

    MLCurlCurl mlcc({geom}, {grids}, {dmap}, info);

    mlcc.setDomainBC({AMREX_D_DECL(LinOpBCType::Periodic,
                                   LinOpBCType::Periodic,
                                   LinOpBCType::Periodic)},
                     {AMREX_D_DECL(LinOpBCType::Periodic,
                                   LinOpBCType::Periodic,
                                   LinOpBCType::Periodic)});

    
    mlcc.setLevelBC(0, &exact);

    mlcc.setScalars(alpha, beta);

    MLMGT<Array<MultiFab,AMREX_SPACEDIM> > mlmg(mlcc);
    mlmg.setMaxIter(max_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    for (auto& mf : solution) {
        mf.setVal(Real(0));
    }
    mlmg.solve({&solution}, {&rhs}, Real(1.0e-10), Real(0));
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("n_cells", n_cells);
    pp.query("max_grid_size", max_grid_size);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("agglomeration", agglomeration);
    pp.query("consolidation", consolidation);
    pp.query("max_coarsening_level", max_coarsening_level);

    pp.query("alpha_over_dx2", alpha_over_dx2);
    pp.query("beta", beta);
}

void
MyTest::initData ()
{
    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain(IntVect(0), IntVect{AMREX_D_DECL(n_cells[0]-1,
                                                n_cells[1]-1,
                                                n_cells[2]-1)});
    geom.define(domain);

    const Real dx = geom.CellSize(0);
    alpha = alpha_over_dx2 * dx*dx;

    grids.define(domain);
    grids.maxSize(max_grid_size);
    dmap.define(grids);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        IntVect itype(1);
        itype[idim] = 0;
        BoxArray const& ba = amrex::convert(grids, itype);
        solution[idim].define(ba,dmap,1,1);
        exact   [idim].define(ba,dmap,1,0);
        rhs     [idim].define(ba,dmap,1,0);
    }

    initProb();

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        exact[idim].LocalCopy(solution[idim], 0, 0, 1, IntVect(0));
    }
}
