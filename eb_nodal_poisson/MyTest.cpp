#include "MyTest.H"

#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_EB2.H>

#include <cmath>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();

    initGrids();

    initializeEB();

    addFineGrids();

    initData();
}

void
MyTest::solve ()
{
    if (verbose > 0) {
        for (int ilev = 0; ilev <= max_level; ++ilev) {
            const MultiFab& vfrc = factory[ilev]->getVolFrac();
            MultiFab v(vfrc.boxArray(), vfrc.DistributionMap(), 1, 0,
                       MFInfo(), *factory[ilev]);
            MultiFab::Copy(v, vfrc, 0, 0, 1, 0);
            amrex::EB_set_covered(v, 1.0);
            amrex::Print() << "Level " << ilev << ": vfrc min = " << v.min(0) << std::endl;
        }
    }

    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (geom[0].isPeriodic(idim)) {
            mlmg_lobc[idim] = LinOpBCType::Periodic;
            mlmg_hibc[idim] = LinOpBCType::Periodic;
        } else {
            mlmg_lobc[idim] = LinOpBCType::Dirichlet;
            mlmg_hibc[idim] = LinOpBCType::Dirichlet;
        }
    }

    LPInfo info;
    info.setMaxCoarseningLevel(max_coarsening_level);
    info.setAgglomerationGridSize(agg_grid_size);
    info.setConsolidationGridSize(con_grid_size);

    static int ipass = 0;
    ++ipass;

    if (composite_solve)
    {
        amrex::Abort("composite_solve not supported");
    }
    else
    {
        for (int ilev = 0; ilev <= max_level; ++ilev)
        {
            BL_PROFILE_REGION("LEVEL-SOLVE-lev"+std::to_string(ilev)+"-pass"+std::to_string(ipass));

            MLEBNodeFDLaplacian mleb({geom[ilev]}, {grids[ilev]}, {dmap[ilev]}, info,
                                     {factory[ilev].get()});

            mleb.setDomainBC(mlmg_lobc, mlmg_hibc);
            phi[ilev].setVal(phi_domain);

            mleb.setSigma({AMREX_D_DECL(1.0, 1.0, 1.0)});
            mleb.setEBDirichlet(phi_eb);

            MLMG mlmg(mleb);
            mlmg.setMaxIter(max_iter);
            mlmg.setMaxFmgIter(max_fmg_iter);
            mlmg.setBottomMaxIter(max_bottom_iter);
            mlmg.setBottomTolerance(bottom_reltol);
            mlmg.setVerbose(verbose);
            mlmg.setBottomVerbose(bottom_verbose);
            if (use_hypre) {
                mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
            } else if (use_petsc) {
                mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
            }

            const Real tol_rel = reltol;
            const Real tol_abs = 0.0;
            mlmg.solve({&phi[ilev]}, {&rhs[ilev]}, tol_rel, tol_abs);
        }
    }
}

void
MyTest::writePlotfile ()
{
    Vector<MultiFab> plotmf(max_level+1);
    for (int ilev = 0; ilev <= max_level; ++ilev) {
        const MultiFab& vfrc = factory[ilev]->getVolFrac();
        amrex::VisMF::Write(vfrc, "vfrc-"+std::to_string(ilev));
        amrex::VisMF::Write(phi[ilev], "phi-"+std::to_string(ilev));
        amrex::VisMF::Write(rhs[ilev], "rhs-"+std::to_string(ilev));
    }
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(max_level == 0, "max_level must be either 0");

    pp.query("phi_domain", phi_domain);
    pp.query("phi_eb", phi_eb);

    pp.query("plot_file", plot_file_name);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_fmg_iter", max_fmg_iter);
    pp.query("max_bottom_iter", max_bottom_iter);
    pp.query("bottom_reltol", bottom_reltol);
    pp.query("reltol", reltol);
    pp.query("max_coarsening_level", max_coarsening_level);
#ifdef AMREX_USE_HYPRE
    pp.query("use_hypre", use_hypre);
#endif
#ifdef AMREX_USE_PETSC
    pp.query("use_petsc", use_petsc);
#endif
    pp.query("agg_grid_size", agg_grid_size);
    pp.query("con_grid_size", con_grid_size);

    pp.query("composite_solve", composite_solve);
}

void
MyTest::initGrids ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);

    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
    std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, isperiodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(ref_ratio);
    }

    // Fine levels will be added later
    grids[0].define(domain0);
    grids[0].maxSize(max_grid_size);
}

void
MyTest::addFineGrids ()
{
    for (int ilev = 1; ilev <= max_level; ++ilev)
    {
        const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
        const EB2::Level& eb_level = eb_is.getLevel(geom[ilev]);
        BoxList bl = eb_level.boxArray().boxList();
        const Box& domain = geom[ilev].Domain();
        for (Box& b : bl) {
            b &= domain;
        }
        grids[ilev].define(bl);
    }
}

void
MyTest::initData ()
{
    int nlevels = max_level + 1;
    dmap.resize(nlevels);
    factory.resize(nlevels);
    phi.resize(nlevels);
    rhs.resize(nlevels);

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
        const EB2::Level& eb_level = eb_is.getLevel(geom[ilev]);
        factory[ilev] = std::make_unique<EBFArrayBoxFactory>
            (eb_level, geom[ilev], grids[ilev], dmap[ilev], Vector<int>{1,1,1}, EBSupport::full);

        BoxArray const& nba = amrex::convert(grids[ilev], IntVect(1));

        phi[ilev].define(nba, dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);
        rhs[ilev].define(nba, dmap[ilev], 1, 0, MFInfo(), *factory[ilev]);

        phi[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);
    }
}
