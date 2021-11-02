#ifndef EM_PIC_K_H_
#define EM_PIC_K_H_

#include "Constants.H"
#include "EMParticleContainer.H"
#include <AMReX_Array4.H>
#include <cmath>

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialized templates are defined below for orders 0 to 3.
 *  Shape factor functors may be evaluated with double arguments
 *  in current deposition to ensure that current deposited by
 *  particles that move only a small distance is still resolved.
 *  Without this safeguard, single and double precision versions
 *  can give disagreeing results in the time evolution for some
 *  problem setups.
 */
template <int depos_order>
struct Compute_shape_factor
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const /*sx*/, T /*xint*/) const { return 0; }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 0
 */
template <>
struct Compute_shape_factor< 0 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const auto j = static_cast<int>(xmid + T(0.5));
        sx[0] = T(1.0);
        return j;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 1
 */
template <>
struct Compute_shape_factor< 1 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const auto j = static_cast<int>(xmid);
        const T xint = xmid - T(j);
        sx[0] = T(1.0) - xint;
        sx[1] = xint;
        return j;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 2
 */
template <>
struct Compute_shape_factor< 2 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const auto j = static_cast<int>(xmid + T(0.5));
        const T xint = xmid - T(j);
        sx[0] = T(0.5)*(T(0.5) - xint)*(T(0.5) - xint);
        sx[1] = T(0.75) - xint*xint;
        sx[2] = T(0.5)*(T(0.5) + xint)*(T(0.5) + xint);
        // index of the leftmost cell where particle deposits
        return j-1;
    }
};

/**
 *  Compute shape factor and return index of leftmost cell where
 *  particle writes.
 *  Specialization for order 3
 */
template <>
struct Compute_shape_factor< 3 >
{
    template< typename T >
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    int operator()(T* const sx, T xmid) const
    {
        const auto j = static_cast<int>(xmid);
        const T xint = xmid - T(j);
        sx[0] = (T(1.0))/(T(6.0))*(T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - xint);
        sx[1] = (T(2.0))/(T(3.0)) - xint*xint*(T(1.0) - xint/(T(2.0)));
        sx[2] = (T(2.0))/(T(3.0)) - (T(1.0) - xint)*(T(1.0) - xint)*(T(1.0) - T(0.5)*(T(1.0) - xint));
        sx[3] = (T(1.0))/(T(6.0))*xint*xint*xint;
        // index of the leftmost cell where particle deposits
        return j-1;
    }
};

/**
 * \brief Field gather for a single particle
 *
 * \tparam depos_order              Particle shape order
 * \tparam galerkin_interpolation   Lower the order of the particle shape by
 *                                  this value (0/1) for the parallel field component
 * \param xp, yp, zp                Particle position coordinates
 * \param Exp, Eyp, Ezp             Electric field on particles.
 * \param Bxp, Byp, Bzp             Magnetic field on particles.
 * \param ex_arr ey_arr ez_arr      Array4 of the electric field, either full array or tile.
 * \param bx_arr by_arr bz_arr      Array4 of the magnetic field, either full array or tile.
 * \param ex_type, ey_type, ez_type IndexType of the electric field
 * \param bx_type, by_type, bz_type IndexType of the magnetic field
 * \param dx                        3D cell spacing
 * \param xyzmin                    Physical lower bounds of box in x, y, z.
 * \param lo                        Index lower bounds of box.
 */
template <int depos_order, int galerkin_interpolation>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void doGatherShapeN (EMParticleContainer::ParticleType const& p,
                     amrex::ParticleReal& Exp,
                     amrex::ParticleReal& Eyp,
                     amrex::ParticleReal& Ezp,
                     amrex::ParticleReal& Bxp,
                     amrex::ParticleReal& Byp,
                     amrex::ParticleReal& Bzp,
                     amrex::Array4<amrex::Real const> const& ex_arr,
                     amrex::Array4<amrex::Real const> const& ey_arr,
                     amrex::Array4<amrex::Real const> const& ez_arr,
                     amrex::Array4<amrex::Real const> const& bx_arr,
                     amrex::Array4<amrex::Real const> const& by_arr,
                     amrex::Array4<amrex::Real const> const& bz_arr,
                     const amrex::GpuArray<amrex::Real, 3>& dx,
                     const amrex::GpuArray<amrex::Real, 3>& xyzmin,
                     const amrex::Dim3& lo)
{
    using namespace amrex;

    const amrex::Real dxi = 1.0/dx[0];
    const amrex::Real dyi = 1.0/dx[1];
    const amrex::Real dzi = 1.0/dx[2];
    
    const amrex::Real xmin = xyzmin[0];
    const amrex::Real ymin = xyzmin[1];
    const amrex::Real zmin = xyzmin[2];

    // --- Compute shape factors
    // x direction
    const amrex::Real x = (p.pos(0)-xmin)*dxi;

    amrex::Real sx_node[depos_order + 1];
    amrex::Real sx_cell_galerkin[depos_order + 1 - galerkin_interpolation] = {0._rt};

    int j_node = 0;
    int j_cell_v = 0;
    Compute_shape_factor<depos_order> const compute_shape_factor;
    Compute_shape_factor<depos_order - galerkin_interpolation> const compute_shape_factor_galerkin;
    j_node = compute_shape_factor(sx_node, x);
    j_cell_v = compute_shape_factor_galerkin(sx_cell_galerkin, x - 0.5_rt);

    const amrex::Real (&sx_ex)[depos_order + 1 - galerkin_interpolation] = sx_cell_galerkin;
    const amrex::Real (&sx_ey)[depos_order + 1                         ] = sx_node;
    const amrex::Real (&sx_ez)[depos_order + 1                         ] = sx_node;
    const amrex::Real (&sx_bx)[depos_order + 1                         ] = sx_node;
    const amrex::Real (&sx_by)[depos_order + 1 - galerkin_interpolation] = sx_cell_galerkin;
    const amrex::Real (&sx_bz)[depos_order + 1 - galerkin_interpolation] = sx_cell_galerkin;
    int const j_ex = j_cell_v;
    int const j_ey = j_node;
    int const j_ez = j_node;
    int const j_bx = j_node;
    int const j_by = j_cell_v;
    int const j_bz = j_cell_v;

    // y direction
    const amrex::Real y = (p.pos(1)-ymin)*dyi;
    amrex::Real sy_node[depos_order + 1];
    amrex::Real sy_cell_v[depos_order + 1 - galerkin_interpolation];
    int k_node = 0;
    int k_cell_v = 0;
    k_node = compute_shape_factor(sy_node, y);
    k_cell_v = compute_shape_factor_galerkin(sy_cell_v, y - 0.5_rt);
    const amrex::Real (&sy_ex)[depos_order + 1                         ] = sy_node;
    const amrex::Real (&sy_ey)[depos_order + 1 - galerkin_interpolation] = sy_cell_v;
    const amrex::Real (&sy_ez)[depos_order + 1                         ] = sy_node;
    const amrex::Real (&sy_bx)[depos_order + 1 - galerkin_interpolation] = sy_cell_v;
    const amrex::Real (&sy_by)[depos_order + 1                         ] = sy_node;
    const amrex::Real (&sy_bz)[depos_order + 1 - galerkin_interpolation] = sy_cell_v;
    int const k_ex = k_node;
    int const k_ey = k_cell_v;
    int const k_ez = k_node;
    int const k_bx = k_cell_v;
    int const k_by = k_node;
    int const k_bz = k_cell_v;

    // z direction
    const amrex::Real z = (p.pos(2)-zmin)*dzi;
    amrex::Real sz_node[depos_order + 1];
    amrex::Real sz_cell_v[depos_order + 1 - galerkin_interpolation];
    int l_node = 0;
    int l_cell_v = 0;
    l_node = compute_shape_factor(sz_node, z);
    l_cell_v = compute_shape_factor_galerkin(sz_cell_v, z - 0.5_rt);
    const amrex::Real (&sz_ex)[depos_order + 1                         ] = sz_node;
    const amrex::Real (&sz_ey)[depos_order + 1                         ] = sz_node;
    const amrex::Real (&sz_ez)[depos_order + 1 - galerkin_interpolation] = sz_cell_v;
    const amrex::Real (&sz_bx)[depos_order + 1 - galerkin_interpolation] = sz_cell_v;
    const amrex::Real (&sz_by)[depos_order + 1 - galerkin_interpolation] = sz_cell_v;
    const amrex::Real (&sz_bz)[depos_order + 1                         ] = sz_node;
    int const l_ex = l_node;
    int const l_ey = l_node;
    int const l_ez = l_cell_v;
    int const l_bx = l_cell_v;
    int const l_by = l_cell_v;
    int const l_bz = l_node;

    // Gather field on particle Exp from field on grid ex_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Exp += sx_ex[ix]*sy_ex[iy]*sz_ex[iz]*
                    ex_arr(lo.x+j_ex+ix, lo.y+k_ex+iy, lo.z+l_ex+iz);
            }
        }
    }
    // Gather field on particle Eyp from field on grid ey_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Eyp += sx_ey[ix]*sy_ey[iy]*sz_ey[iz]*
                    ey_arr(lo.x+j_ey+ix, lo.y+k_ey+iy, lo.z+l_ey+iz);
            }
        }
    }
    // Gather field on particle Ezp from field on grid ez_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Ezp += sx_ez[ix]*sy_ez[iy]*sz_ez[iz]*
                    ez_arr(lo.x+j_ez+ix, lo.y+k_ez+iy, lo.z+l_ez+iz);
            }
        }
    }
    // Gather field on particle Bzp from field on grid bz_arr
    for (int iz=0; iz<=depos_order; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Bzp += sx_bz[ix]*sy_bz[iy]*sz_bz[iz]*
                    bz_arr(lo.x+j_bz+ix, lo.y+k_bz+iy, lo.z+l_bz+iz);
            }
        }
    }
    // Gather field on particle Byp from field on grid by_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<=depos_order; iy++){
            for (int ix=0; ix<= depos_order - galerkin_interpolation; ix++){
                Byp += sx_by[ix]*sy_by[iy]*sz_by[iz]*
                    by_arr(lo.x+j_by+ix, lo.y+k_by+iy, lo.z+l_by+iz);
            }
        }
    }
    // Gather field on particle Bxp from field on grid bx_arr
    for (int iz=0; iz<= depos_order - galerkin_interpolation; iz++){
        for (int iy=0; iy<= depos_order - galerkin_interpolation; iy++){
            for (int ix=0; ix<=depos_order; ix++){
                Bxp += sx_bx[ix]*sy_bx[iy]*sz_bx[iz]*
                    bx_arr(lo.x+j_bx+ix, lo.y+k_bx+iy, lo.z+l_bx+iz);
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void push_momentum_boris (amrex::Real& uxp, amrex::Real& uyp, amrex::Real& uzp, amrex::Real& gaminv,
                          amrex::Real Exp, amrex::Real Eyp, amrex::Real Ezp,
                          amrex::Real Bxp, amrex::Real Byp, amrex::Real Bzp,
                          amrex::Real q, amrex::Real m, amrex::Real dt)
{
    amrex::Real cst = q*dt*0.5/m;
    constexpr amrex::Real clight = 2.99792458e8;
    constexpr amrex::Real clightisq = 1./(clight*clight);

    uxp += Exp * cst;
    uyp += Eyp * cst;
    uzp += Ezp * cst;

    amrex::Real usq = (uxp*uxp + uyp*uyp + uzp*uzp) * clightisq;
    amrex::Real gaminvtmp = 1.0/std::sqrt(1.0+usq);

    amrex::Real tx = gaminvtmp * Bxp * cst;
    amrex::Real ty = gaminvtmp * Byp * cst;
    amrex::Real tz = gaminvtmp * Bzp * cst;
    amrex::Real tsqi = 2.0/(1.0 + tx*tx + ty*ty + tz*tz);
    amrex::Real sx = tx*tsqi;
    amrex::Real sy = ty*tsqi;
    amrex::Real sz = tz*tsqi;
    amrex::Real uxppr = uxp + uyp*tz - uzp*ty;
    amrex::Real uyppr = uyp + uzp*tx - uxp*tz;
    amrex::Real uzppr = uzp + uxp*ty - uyp*tx;
    uxp += uyppr*sz - uzppr*sy;
    uyp += uzppr*sx - uxppr*sz;
    uzp += uxppr*sy - uyppr*sx;

    uxp += Exp*cst;
    uyp += Eyp*cst;
    uzp += Ezp*cst;

    usq = (uxp*uxp + uyp*uyp + uzp*uzp) * clightisq;
    gaminv = 1.0/std::sqrt(1.0+usq);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void push_position_boris (EMParticleContainer::ParticleType& p,
                          amrex::Real uxp, amrex::Real uyp, amrex::Real uzp,
                          amrex::Real gaminv, amrex::Real dt)
{
    p.pos(0) += uxp*gaminv*dt;
    p.pos(1) += uyp*gaminv*dt;
    p.pos(2) += uzp*gaminv*dt;
}

#endif
