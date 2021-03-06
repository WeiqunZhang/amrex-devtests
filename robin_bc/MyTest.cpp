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
    MLABecLaplacian mlabec({geom}, {grids}, {dmap},
                           LPInfo().setMaxCoarseningLevel(max_coarsening_level));

    // This is a 3d problem with Dirichlet BC
    Array<LinOpBCType,AMREX_SPACEDIM> lobc{AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                        LinOpBCType::Dirichlet,
                                                        LinOpBCType::Dirichlet)};
    Array<LinOpBCType,AMREX_SPACEDIM> hibc{AMREX_D_DECL(LinOpBCType::Dirichlet,
                                                        LinOpBCType::Dirichlet,
                                                        LinOpBCType::Dirichlet)};
    if (robin_face == 0) {
        lobc[robin_dir] = LinOpBCType::Robin;
    } else {
        hibc[robin_dir] = LinOpBCType::Robin;
    }
    mlabec.setDomainBC(lobc,hibc);

    mlabec.setLevelBC(0, &solution, &robin_a, &robin_b, &robin_f);

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
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("robin_dir", robin_dir);
    pp.query("robin_face", robin_face);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);
    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("max_coarsening_level", max_coarsening_level);
}

void
MyTest::initData ()
{
    RealBox rb({AMREX_D_DECL(0.3,0.3,0.3)}, {AMREX_D_DECL(1.6,1.6,1.6)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    geom.define(domain0);

    grids.define(domain0);
    grids.maxSize(max_grid_size);

    IntVect ng = IntVect{1};

    dmap.define(grids);
    solution      .define(grids, dmap, 1, ng);
    rhs           .define(grids, dmap, 1, 0);
    exact_solution.define(grids, dmap, 1, ng);
    acoef.define(grids, dmap, 1, 0);
    bcoef.define(grids, dmap, 1, ng);
    robin_a.define(grids, dmap, 1, ng);
    robin_b.define(grids, dmap, 1, ng);
    robin_f.define(grids, dmap, 1, ng);

    initProbABecLaplacian();
}
