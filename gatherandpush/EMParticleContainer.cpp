#include "EMParticleContainer.H"
#include "Constants.H"

#include "em_pic_K.H"

using namespace amrex;

amrex::GpuArray<amrex::Real, 3> LowerCorner (const Box& bx, const Geometry& gm)
{
    RealBox grid_box{bx, gm.CellSize(), gm.ProbLo()};
    const Real* xyzmin = grid_box.lo();
    return { xyzmin[0], xyzmin[1], xyzmin[2]};
}

void EMParticleContainer::
GatherAndPush(const MultiFab& Ex, const MultiFab& Ey, const MultiFab& Ez,
              const MultiFab& Bx, const MultiFab& By, const MultiFab& Bz, Real dt)
{
    BL_PROFILE("EMParticleContainer::GatherAndPush");

    const int lev = 0;
    const auto dx = Geom(lev).CellSizeArray();
    const auto plo = Geom(lev).ProbLoArray();
    for (EMParIter pti(*this, lev); pti.isValid(); ++pti)
    {
        const int np  = pti.numParticles();
        const Box& box = pti.tilebox();
        const auto& xyzmin = LowerCorner(box, Geom(lev));
        const Dim3 lo = lbound(box);

        ParticleType * pstruct = &(pti.GetArrayOfStructs()[0]);

        auto& attribs = pti.GetAttribs();
        Real *  wp  = attribs[PIdx::w].data();
        Real * uxp  = attribs[PIdx::ux].data();
        Real * uyp  = attribs[PIdx::uy].data();
        Real * uzp  = attribs[PIdx::uz].data();

        auto const Exarr = Ex.array(pti);
        auto const Eyarr = Ey.array(pti);
        auto const Ezarr = Ez.array(pti);
        auto const Bxarr = Bx.array(pti);
        auto const Byarr = By.array(pti);
        auto const Bzarr = Bz.array(pti);

        Real q = m_charge;
        Real m = m_mass;
        amrex::ParallelFor( np,
        [=] AMREX_GPU_DEVICE (int i) noexcept                            
        {
            amrex::Real Exp = 0.0;
            amrex::Real Eyp = 0.0;
            amrex::Real Ezp = 0.0;
            amrex::Real Bxp = 0.0;
            amrex::Real Byp = 0.0;
            amrex::Real Bzp = 0.0;

            doGatherShapeN<3, 1>(pstruct[i], Exp, Eyp, Ezp, Bxp, Byp, Bzp,
                                 Exarr, Eyarr, Ezarr, Bxarr, Byarr, Bzarr, dx, xyzmin, lo);

            amrex::Real ginv = 0.0;
            push_momentum_boris(uxp[i], uyp[i], uzp[i], ginv, Exp, Eyp, Ezp,
                                Bxp, Byp, Bzp, q, m, dt);

            push_position_boris(pstruct[i], uxp[i], uyp[i], uzp[i], ginv, dt);
        });
    }
}
