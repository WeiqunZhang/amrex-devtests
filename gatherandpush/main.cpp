#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>

#include "EMParticleContainer.H"
#include "Constants.H"

using namespace amrex;

namespace YeeGrid {
    static constexpr IntVect Bx_nodal_flag(1,0,0);
    static constexpr IntVect By_nodal_flag(0,1,0);
    static constexpr IntVect Bz_nodal_flag(0,0,1);

    static constexpr IntVect Ex_nodal_flag(0,1,1);
    static constexpr IntVect Ey_nodal_flag(1,0,1);
    static constexpr IntVect Ez_nodal_flag(1,1,0);
};

Real compute_dt (const Geometry& geom)
{
    const static Real cfl = 1.0;
    const Real* dx = geom.CellSize();
    const Real dt  = cfl * 1./( std::sqrt(D_TERM(  1./(dx[0]*dx[0]),
                                                 + 1./(dx[1]*dx[1]),
                                                 + 1./(dx[2]*dx[2]))) * PhysConst::c );
    return dt;
}

struct TestParams
{
    IntVect ncell;
    IntVect nppc;
    int max_grid_size;
};

void test_em_pic (const TestParams& parms)
{
    BL_PROFILE("test_em_pic");

    RealBox real_box;
    for (int n = 0; n < BL_SPACEDIM; n++)
    {
        real_box.setLo(n, -20e-6);
        real_box.setHi(n,  20e-6);
    }

    IntVect domain_lo(AMREX_D_DECL(0, 0, 0));
    IntVect domain_hi(AMREX_D_DECL(parms.ncell[0]-1,parms.ncell[1]-1,parms.ncell[2]-1));
    const Box domain(domain_lo, domain_hi);

    int coord = 0;
    int is_per[BL_SPACEDIM];
    for (int i = 0; i < BL_SPACEDIM; i++)
        is_per[i] = 1;
    Geometry geom(domain, &real_box, coord, is_per);

    BoxArray ba(domain);
    ba.maxSize(parms.max_grid_size);
    DistributionMapping dm(ba);

    const int ng = 3;
    MultiFab Bx(amrex::convert(ba, YeeGrid::Bx_nodal_flag), dm, 1, ng);
    MultiFab By(amrex::convert(ba, YeeGrid::By_nodal_flag), dm, 1, ng);
    MultiFab Bz(amrex::convert(ba, YeeGrid::Bz_nodal_flag), dm, 1, ng);

    MultiFab Ex(amrex::convert(ba, YeeGrid::Ex_nodal_flag), dm, 1, ng);
    MultiFab Ey(amrex::convert(ba, YeeGrid::Ey_nodal_flag), dm, 1, ng);
    MultiFab Ez(amrex::convert(ba, YeeGrid::Ez_nodal_flag), dm, 1, ng);

    Ex.setVal(0.0); Ey.setVal(0.0); Ez.setVal(0.0);
    Bx.setVal(0.0); By.setVal(0.0); Bz.setVal(0.0);

    EMParticleContainer particles(geom, dm, ba, -PhysConst::q_e, PhysConst::m_e);
    particles.InitParticles(parms.nppc, 0.01, 10.0, 1e25, real_box);

    const Real dt = compute_dt(geom);
    particles.GatherAndPush(Ex, Ey, Ez, Bx, By, Bz, dt);
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        ParmParse pp;
        TestParams parms;

        pp.get("ncell", parms.ncell);
        pp.get("nppc",  parms.nppc);
        pp.get("max_grid_size", parms.max_grid_size);

        test_em_pic(parms);
    }
    amrex::Finalize();
}
