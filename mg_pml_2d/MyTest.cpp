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

    initData();
}

void
MyTest::solve ()
{
    if (method == 0) {
        solve0();
    } else if (method == 1) {
        solve1();
    } else if (method == 2) {
        solve2();
    } else {
        amrex::Abort("Unknown method");
    }
}

void
MyTest::solve2 ()
{
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
    mlmg_lobc[0] = LinOpBCType::Dirichlet;
    mlmg_hibc[0] = LinOpBCType::Dirichlet;
    mlmg_lobc[1] = LinOpBCType::Dirichlet;
    mlmg_hibc[1] = LinOpBCType::Dirichlet;

    LPInfo info;
    info.setMaxCoarseningLevel(0);

    MLEBNodeFDLaplacian mleb({geom[0]}, {grids[0]}, {dmap[0]}, info,
                             {factory[0].get()});
    mleb.setDomainBC(mlmg_lobc, mlmg_hibc);

    mleb.setEBDirichlet([=] AMREX_GPU_DEVICE (Real x, Real) {
        return (x < 0) ? Real(1) : Real(-1);
    });

    auto const& ea = exact[0].const_arrays();
    auto const& pa = phi[0].arrays();
    ParallelFor(phi[0], [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
    {
        if (i != 0 && i != 160 && j != 0 && j != 160) {
            pa[b](i,j,k) = ea[b](i,j,k);
        }
    });

    MultiFab sigma(amrex::convert(grids[0],IntVect(0)), dmap[0], 1, 2);
    sigma.setVal(1.0);
    mleb.setSigma(0,sigma);

    MLMG mlmg(mleb);
    mlmg.setMaxIter(max_iter);
    mlmg.setBottomMaxIter(max_bottom_iter);
    mlmg.setBottomTolerance(bottom_reltol);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    const Real tol_rel = reltol;
    const Real tol_abs = 0.0;
    mlmg.solve({&phi[0]}, {&rhs[0]}, tol_rel, tol_abs);

    VisMF::Write(phi[0], "phi-big-2");
}

void
MyTest::solve1 ()
{
    solve0();
}

void
MyTest::solve0 ()
{
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
    mlmg_lobc[0] = LinOpBCType::Dirichlet;
    mlmg_hibc[0] = LinOpBCType::Dirichlet;
    mlmg_lobc[1] = LinOpBCType::Dirichlet;
    mlmg_hibc[1] = LinOpBCType::Dirichlet;

    MLEBNodeFDLaplacian mleb({geom[0]}, {grids[0]}, {dmap[0]}, LPInfo{},
                             {factory[0].get()});
    mleb.setDomainBC(mlmg_lobc, mlmg_hibc);

    mleb.setEBDirichlet([=] AMREX_GPU_DEVICE (Real x, Real) {
        return (x < 0) ? Real(1) : Real(-1);
    });

    MLMG mlmg(mleb);
    mlmg.setMaxIter(max_iter);
    mlmg.setBottomMaxIter(max_bottom_iter);
    mlmg.setBottomTolerance(bottom_reltol);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);

    MultiFab::Copy(phi[0], exact[0], 0, 0, 1, 0);

    const Real tol_rel = reltol;
    const Real tol_abs = 0.0;
    mlmg.solve({&phi[0]}, {&rhs[0]}, tol_rel, tol_abs);
}

void
MyTest::writePlotfile ()
{
    MultiFab phi_tmp, exact_tmp;
    if (method == 0) {
        phi_tmp = MultiFab(phi[0], amrex::make_alias, 0, 1);
        exact_tmp = MultiFab(exact[0], amrex::make_alias, 0, 1);
    } else {
        Box domain = geom[0].Domain();
        domain.grow(-n_cell_pml);
        BoxArray ba(domain);
        ba.maxSize(max_grid_size);
        ba.convert(IntVect(1));
        DistributionMapping dm{ba};
        phi_tmp.define(ba, dm, 1, 0);
        exact_tmp.define(ba, dm, 1, 0);
        phi_tmp.ParallelCopy(phi[0]);
        exact_tmp.ParallelCopy(exact[0]);
    }
    MultiFab error(phi_tmp.boxArray(), phi_tmp.DistributionMap(), 1, 0);
    MultiFab::Copy(error, phi_tmp, 0, 0, 1, 0);
    MultiFab::Subtract(error, exact_tmp, 0, 0, 1, 0);
    amrex::Print() << "xxxx error : " << error.min(0) << ", " << error.max(0) << "\n";
    amrex::VisMF::Write(phi_tmp, "phi-"+std::to_string(method));
    amrex::VisMF::Write(exact_tmp, "exact-"+std::to_string(method));
    amrex::VisMF::Write(error, "error-"+std::to_string(method));
}

void
MyTest::readParameters ()
{
    ParmParse pp;

    pp.query("method", method);
    pp.queryAdd("n_cell_pml", n_cell_pml);
    pp.query("prob.a", prob_a);
    pp.query("prob.d", prob_d);
    AMREX_ALWAYS_ASSERT(prob_d > 2*prob_a);

    pp.queryAdd("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    if (method == 2) { max_grid_size = 8*n_cell; }

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_iter", max_iter);
    pp.query("max_bottom_iter", max_bottom_iter);
    pp.query("bottom_reltol", bottom_reltol);
    pp.query("reltol", reltol);
}

void
MyTest::initGrids ()
{
    int nlevels = 1;
    geom.resize(nlevels);
    grids.resize(nlevels);

    RealBox rb({-0.5,-0.5}, {0.5,0.5});
    if (method > 0) {
        auto dx_extra =  Real(n_cell_pml)/ Real(n_cell);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            rb.setLo(idim, rb.lo(idim)-dx_extra);
            rb.setHi(idim, rb.hi(idim)+dx_extra);
        }
    }
    std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,0,0)};
    int n_extra = (method == 0) ? 0 : n_cell_pml*2;
    int ntot = n_cell + n_extra;
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(ntot-1,ntot-1,ntot-1)});
    geom[0].define(domain0, &rb, 0, isperiodic.data());

    amrex::Print() << "xxxx geom[0] = " << geom[0] << std::endl;

    // Fine levels will be added later
    grids[0].define(domain0);
    grids[0].maxSize(max_grid_size);
}

void
MyTest::initData ()
{
    int nlevels = 1;
    dmap.resize(nlevels);
    factory.resize(nlevels);
    phi.resize(nlevels);
    exact.resize(nlevels);
    rhs.resize(nlevels);

    dmap[0].define(grids[0]);
    const EB2::IndexSpace& eb_is = EB2::IndexSpace::top();
    const EB2::Level& eb_level = eb_is.getLevel(geom[0]);
    factory[0] = std::make_unique<EBFArrayBoxFactory>
        (eb_level, geom[0], grids[0], dmap[0], Vector<int>{1,1,1}, EBSupport::full);

    BoxArray const& nba = amrex::convert(grids[0], IntVect(1));

    phi[0].define(nba, dmap[0], 1, 0, MFInfo(), *factory[0]);
    exact[0].define(nba, dmap[0], 1, 0, MFInfo(), *factory[0]);
    rhs[0].define(nba, dmap[0], 1, 0, MFInfo(), *factory[0]);

    phi[0].setVal(0);
    rhs[0].setVal(0);

    Real c = 0.5*std::sqrt(prob_d*prob_d - 4.*prob_a*prob_a);
    auto tmp = prob_d/(2*prob_a);
    Real mu0 = std::log(tmp + std::sqrt(tmp*tmp-1));

    auto const& problo = geom[0].ProbLoArray();
    auto const& dx     = geom[0].CellSizeArray();

    auto const& levset = factory[0]->getLevelSet();
    for (MFIter mfi(exact[0]); mfi.isValid(); ++mfi) {
        auto const& box = mfi.validbox();
        auto const& a = exact[0].array(mfi);
        auto const& lsa =levset.const_array(mfi);
        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            auto x = Real(i) * dx[0] + problo[0];
            auto y = Real(j) * dx[1] + problo[1];
            if (lsa(i,j,0) >= 0) {
                a(i,j,0) = (x < 0) ? Real(1.0) : Real(-1.0);
            } else {
                auto mu = std::log(std::sqrt((x-c)*(x-c)+y*y)
                                   /std::sqrt((x+c)*(x+c)+y*y));
                a(i,j,0) = mu/mu0;
            }
        });
    }
}
