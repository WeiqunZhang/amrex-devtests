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
    if (prob_type == 1) {
        solvePoisson();
    } else if (prob_type == 2) {
        solveABecLaplacian();
    } else if (prob_type == 3) {
        solveALaplacian();
    } else {
        amrex::Abort("Unknown prob_type");
    }
}

void
MyTest::solvePoisson ()
{
    LPInfo lpinfo;
    lpinfo.setMaxCoarseningLevel(max_coarsening_level).setHiddenDirection(hidden_direction);
    MLPoisson mlpoisson({geom}, {grids}, {dmap}, lpinfo);

    // This is a 3d problem with Dirichlet BC
    mlpoisson.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet)},
                          {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet,
                                        LinOpBCType::Dirichlet)});

    mlpoisson.setLevelBC(0, &solution);

    MLMG mlmg(mlpoisson);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
}

void
MyTest::solveABecLaplacian ()
{
    MLABecLaplacian mlabec({geom}, {grids}, {dmap},
                           LPInfo().setMaxCoarseningLevel(max_coarsening_level));

    // This is a 3d problem with Dirichlet BC
    mlabec.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet)},
                       {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet,
                                     LinOpBCType::Dirichlet)});

    mlabec.setLevelBC(0, nullptr);

    mlabec.setScalars(ascalar, bscalar);

    mlabec.setACoeffs(0, acoef);

    Array<MultiFab,AMREX_SPACEDIM> face_bcoef;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        const BoxArray& ba = amrex::convert(bcoef.boxArray(),
                                            IntVect::TheDimensionVector(idim));
        face_bcoef[idim].define(ba, bcoef.DistributionMap(), 1, 0);
    }
    amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoef),
                                      bcoef, geom);
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(face_bcoef));

    MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);            
}

void
MyTest::solveALaplacian ()
{
    LPInfo lpinfo;
    lpinfo.setMaxCoarseningLevel(max_coarsening_level).setHiddenDirection(hidden_direction);
    MLALaplacian mlalaplacian({geom}, {grids}, {dmap}, lpinfo);

    // This is a 3d problem with Dirichlet BC
    mlalaplacian.setDomainBC({AMREX_D_DECL(LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet)},
                            {AMREX_D_DECL(LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet,
                                          LinOpBCType::Dirichlet)});

    mlalaplacian.setLevelBC(0, &solution);

    mlalaplacian.setScalars(ascalar, bscalar);

    mlalaplacian.setACoeffs(0, acoef);

    MLMG mlmg(mlalaplacian);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    amrex::Print() << "xxxxx" << std::endl;

    const Real tol_rel = 1.e-10;
    const Real tol_abs = 0.0;
    mlmg.solve({&solution}, {&rhs}, tol_rel, tol_abs);
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.query("hidden_direction", hidden_direction);

    pp.query("prob_type", prob_type);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("max_coarsening_level", max_coarsening_level);
}

void
MyTest::initData ()
{
    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.3,1.3,1.3)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    geom.define(domain0);

    domain0.setRange(hidden_direction, n_cell/2);
    grids.define(domain0);
    grids.maxSize(max_grid_size);

    IntVect ng = IntVect{1};
    ng[hidden_direction] = 0;

    dmap.define(grids);
    solution      .define(grids, dmap, 1, ng);
    rhs           .define(grids, dmap, 1, 0);
    exact_solution.define(grids, dmap, 1, ng);
    if (prob_type != 1) {
        acoef.define(grids, dmap, 1, 0);
    }
    if (prob_type == 2) {
        bcoef.define(grids, dmap, 1, ng);
    }

    if (prob_type == 1) {
        initProbPoisson();
//    } else if (prob_type == 2) {
//        initProbABecLaplacian();
    } else if (prob_type == 3) {
        initProbALaplacian();
    } else {
        amrex::Abort("Unsupported prob_type "+std::to_string(prob_type));
    }
}

