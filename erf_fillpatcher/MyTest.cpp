#include "MyTest.H"
#include <ERF_FillPatcher.H>

#include <AMReX_ParmParse.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();
    initData();
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("max_level", max_level);
    pp.query("ref_ratio", ref_ratio);
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);
}

void
MyTest::initData ()
{
    int nlevels = max_level + 1;
    geom.resize(nlevels);
    grids.resize(nlevels);
    dmap.resize(nlevels);

    phi.resize(nlevels);

    RealBox rb({AMREX_D_DECL(0.,0.,0.)}, {AMREX_D_DECL(1.,1.,1.)});
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(0,0,1)};
    Geometry::Setup(&rb, 0, is_periodic.data());
    Box domain0(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
    Box domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        geom[ilev].define(domain);
        domain.refine(ref_ratio);
    }

    domain = domain0;
    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        grids[ilev].define(domain);
        grids[ilev].maxSize(max_grid_size);
        domain.grow(IntVect(-n_cell/4,-n_cell/4,0)); // fine level cover the middle of the coarse domain
        domain.refine(ref_ratio);
    }

    for (int ilev = 0; ilev < nlevels; ++ilev)
    {
        dmap[ilev].define(grids[ilev]);
        phi[ilev].define(grids[ilev], dmap[ilev], 1, 1);
    }

    AMREX_ALWAYS_ASSERT(max_level == 1);
    const auto problo = geom[0].ProbLoArray();
    const auto dx     = geom[0].CellSizeArray();
    auto const& a = phi[0].arrays();
    amrex::ParallelFor(phi[0], phi[0].nGrowVect(),
    [=] AMREX_GPU_DEVICE (int bi, int i, int j, int k) noexcept
    {
        constexpr Real pi = 3.1415926535897932;
        constexpr Real tpi = 2.*pi;
        constexpr Real fpi = 4.*pi;
        amrex::Real x = problo[0] + dx[0] * (i + 0.5);
        amrex::Real y = problo[1] + dx[1] * (j + 0.5);
        amrex::Real z = problo[2] + dx[2] * (k + 0.5);
        a[bi](i,j,k) = (std::sin(tpi*x) * std::sin(tpi*y) * std::sin(tpi*z))
            +    .25 * (std::sin(fpi*x) * std::sin(fpi*y) * std::sin(fpi*z));
    });
    phi[1].setVal(0.0);
}

void
MyTest::test ()
{
    int nghost = -2;
    int ncomp = 1;
    auto interp = &cell_cons_interp;
    ERFFillPatcher erffp(grids[1], dmap[1], geom[1], grids[0], dmap[0], geom[0],
                         nghost, ncomp, interp); // This can be reused..

    // Assuming it's periodic in z-direction
    Vector<BCRec> bcr(1);
    bcr[0].setLo(0, BCType::ext_dir);
    bcr[0].setHi(0, BCType::ext_dir);
    bcr[0].setLo(1, BCType::ext_dir);
    bcr[0].setHi(1, BCType::ext_dir);
    bcr[0].setLo(2, BCType::int_dir);
    bcr[0].setHi(2, BCType::int_dir);
    PhysBCFunctNoOp cbc{};

    for (int i = 0; i < 3; ++i) {
        phi[0].plus(1.0,0);
        erffp.registerCoarseData({&phi[0]}, {0.0});
        erffp.fillCoarseFineBoundary(phi[1], 0.0, cbc, bcr);
    }
}
