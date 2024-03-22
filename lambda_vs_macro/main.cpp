#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

enum class HOScheme {
    TVD
};

enum class TVDScheme
{
    UMIST,
};

template<typename T>
[[nodiscard]]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T upwind_scheme (int i, int j, int, int n,
                    amrex::Array4<T const> const& uface_x,
                    amrex::Array4<T const> const& uface_y,
                    amrex::Array4<T const> const& u)
{
    constexpr T zero = T(0.0);
    const T upwind_hi_x = amrex::max(uface_x(i+1, j, 0), zero) * u(i  , j, 0, n) +
                          amrex::min(uface_x(i+1, j, 0), zero) * u(i+1, j, 0, n);
    const T upwind_lo_x = amrex::max(uface_x(i, j, 0), zero) * u(i-1, j, 0, n) +
                          amrex::min(uface_x(i, j, 0), zero) * u(i  , j, 0, n);
    const T upwind_term_x = upwind_hi_x - upwind_lo_x;

    const T upwind_hi_y = amrex::max(uface_y(i, j+1, 0), zero) * u(i, j  , 0, n) +
                          amrex::min(uface_y(i, j+1, 0), zero) * u(i, j+1, 0, n);
    const T upwind_lo_y = amrex::max(uface_y(i, j, 0), zero) * u(i, j-1, 0, n) +
                          amrex::min(uface_y(i, j, 0), zero) * u(i, j  , 0, n);
    const T upwind_term_y = upwind_hi_y - upwind_lo_y;

    return upwind_term_x + upwind_term_y;
}

#define EPSILON 1e-13

// None case
template<HOScheme hos, TVDScheme tvds>
struct limiter { };


// UMIST case
template<>
struct limiter<HOScheme::TVD, TVDScheme::UMIST> {
    template<typename T>
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    constexpr T operator()(T num, T den) noexcept {
        den += EPSILON;
        const T r = num / den;
        using RT = amrex::Real;
        return den * amrex::max<RT>(0, amrex::min<RT>(2, 2 * r, 0.25 * (1 + 3.0 * r), 0.25 * (3.0 + r)));
    }
};


template<HOScheme hos, TVDScheme tvds>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
constexpr limiter<hos, tvds> getScheme() noexcept
{
    return limiter<hos, tvds>{};
}

template<HOScheme hos, TVDScheme tvds, typename T>
[[nodiscard]]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T tvd_scheme (int i, int j, int k, int n,
                amrex::Array4<T const> const& uface_x,
                amrex::Array4<T const> const& uface_y,
                amrex::Array4<T const> const& u) noexcept
{
    constexpr T zero = T(0.0);

    auto limiter = getScheme<hos, tvds>();

    const T tvd_hi_x = amrex::max(uface_x(i+1, j, k), zero) * (
                            u(i  , j, k, n) + 0.5 * limiter(u(i  , j, k, n) - u(i-1, j, k, n), u(i+1, j, k, n) - u(i  , j, k, n))
                        )
                        +
                        amrex::min(uface_x(i+1, j, k), zero) * (
                            u(i+1, j, k, n) + 0.5 * limiter(u(i+1, j, k, n) - u(i+2, j, k, n), u(i  , j, k, n) - u(i+1, j, k, n))
                        );

    const T tvd_lo_x = amrex::max(uface_x(i, j, k), zero) * (
                            u(i-1, j, k, n) + 0.5 * limiter(u(i-1, j, k, n) - u(i-2, j, k, n), u(i  , j, k, n) - u(i-1, j, k, n))
                        )
                        +
                        amrex::min(uface_x(i, j, k), zero) * (
                            u(i, j, k, n) + 0.5 * limiter(u(i  , j, k, n) - u(i+1, j, k, n), u(i-1, j, k, n) - u(i  , j, k, n))
                        );

    const T tvd_term_x = tvd_hi_x - tvd_lo_x;

    const T tvd_hi_y = amrex::max(uface_y(i, j+1, k), zero) * (
                            u(i, j  , k, n) + 0.5 * limiter(u(i, j  , k, n) - u(i, j-1, k, n), u(i, j+1, k, n) - u(i, j  , k, n))
                        )
                        +
                        amrex::min(uface_y(i, j+1, k), zero) * (
                            u(i, j+1, k, n) + 0.5 * limiter(u(i, j+1, k, n) - u(i, j+2, k, n), u(i, j  , k, n) - u(i, j+1, k, n))
                        );

    const T tvd_lo_y = amrex::max(uface_y(i, j, k), zero) * (
                            u(i, j-1, k, n) + 0.5 * limiter(u(i, j-1, k, n) - u(i, j-2, k, n), u(i, j  , k, n) - u(i, j-1, k, n))
                        )
                        +
                        amrex::min(uface_y(i, j, k), zero) * (
                            u(i, j, k, n) + 0.5 * limiter(u(i, j  , k, n) - u(i, j+1, k, n), u(i, j-1, k, n) - u(i, j  , k, n))
                        );
    const T tvd_term_y = tvd_hi_y - tvd_lo_y;

    return tvd_term_x + tvd_term_y;
}

namespace {

namespace test{

template<HOScheme hos, TVDScheme tvds, typename T>
struct deferred_correction {
    [[nodiscard]]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    static
    T impl (int i, int j, int k, int n,
                amrex::Array4<T const> const& uface_x,
                amrex::Array4<T const> const& uface_y,
                amrex::Array4<T const> const& u,
                amrex::GpuArray<T,AMREX_SPACEDIM> const& dxinv,
                T blend_factor) noexcept
    {
        // static_assert(hos == HOScheme::TVD, "Invalid combination of schemes");
        const T ho_term = tvd_scheme<hos, tvds>(i, j, k, n, uface_x, uface_y, u);
        const T upwind_term  = upwind_scheme (i, j, k, n, uface_x, uface_y, u);

        return blend_factor * dxinv[n] * (upwind_term - ho_term);
    }
};

} // namespace detail
} // End anonymous namespace

template <HOScheme hos, TVDScheme tvds, typename T>
[[nodiscard]]
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
T deferred_correction (int i, int j, int k, int n,
                    amrex::Array4<T const> const& uface_x,
                    amrex::Array4<T const> const& uface_y,
                    amrex::Array4<T const> const& u,
                    amrex::GpuArray<T,AMREX_SPACEDIM> const& dxinv,
                    T blend_factor) noexcept
{
    return test::deferred_correction<hos, tvds, T>::impl(i, j, k, n, uface_x, uface_y, u, dxinv, blend_factor);
}

#undef EPSILON


void testParallelFor()
{

    int n_cell = 512;
    int max_grid_size = 32;

    Geometry geom;
    BoxArray grids;
    DistributionMapping dmap;
    {
        RealBox rb ({AMREX_D_DECL(0.0, 0.0, 0.0)}, {AMREX_D_DECL(1.0, 1.0, 1.0)});
        Array<int,AMREX_SPACEDIM> is_periodic {AMREX_D_DECL(0,0,0)}; // Not periodic
        Geometry::Setup(&rb, 0, is_periodic.data());
        Box domain(IntVect(AMREX_D_DECL(0,0,0)), IntVect(AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)));
        geom.define(domain);

        grids.define(domain);
        grids.maxSize(max_grid_size);
        dmap.define(grids);
    }

    MultiFab vel;
    MultiFab rhs;
    MultiFab gp;
    MultiFab Usrc;
    Array<MultiFab,AMREX_SPACEDIM> velface;

    // Fill vel, rhs, gp, velface with random values
    {
        vel.define(grids, dmap, AMREX_SPACEDIM, 2);
        rhs.define(grids, dmap, AMREX_SPACEDIM, 0);
        gp.define(grids, dmap, AMREX_SPACEDIM, 0);
        Usrc.define(grids, dmap, AMREX_SPACEDIM, 0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            velface[idim].define(amrex::convert(grids, IntVect::TheDimensionVector(idim)), dmap, 1, 0);
        }

        vel.setVal(0.0);
        rhs.setVal(0.0);
        gp.setVal(0.0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            velface[idim].setVal(0.0);
        }

        for (MFIter mfi(vel); mfi.isValid(); ++mfi) {
            const Box& bx = mfi.validbox();
            auto const& vel_arr = vel.array(mfi);
            auto const& rhs_arr = rhs.array(mfi);
            auto const& gp_arr = gp.array(mfi);
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                auto const& velface_arr = velface[idim].array(mfi);
                amrex::ParallelFor(bx, AMREX_SPACEDIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        vel_arr(i,j,k,n) = amrex::Random();
                        rhs_arr(i,j,k,n) = amrex::Random();
                        gp_arr(i,j,k,n) = amrex::Random();
                        velface_arr(i,j,k) = amrex::Random();
                    }
                );
            }
        }
    }

    {
        BL_PROFILE("lambda");

        for (int i = 0; i < 100; ++i) {
            const auto& dxinv = geom.InvCellSizeArray();
            const Real blend_factor = 0.5;
            AMREX_D_TERM(auto& uxface = velface[0];,
                         auto& uyface = velface[1];,
                         auto& uzface = velface[2];);

            MFItInfo mfi_info;
            if (Gpu::notInLaunchRegion()) { mfi_info.EnableTiling().SetDynamic(true); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(vel, mfi_info); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                const auto& vel_fab = vel.const_array(mfi);
                const auto& gp_fab = gp.const_array(mfi);
                const auto& Usrc_fab = Usrc.array(mfi);

                AMREX_D_TERM(const auto& uxface_fab = uxface.const_array(mfi);,
                             const auto& uyface_fab = uyface.const_array(mfi);,
                             const auto& uzface_fab = uzface.const_array(mfi););
                amrex::ParallelFor(
                    bx, AMREX_SPACEDIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        amrex::Real deferred_correction_term = deferred_correction<HOScheme::TVD, TVDScheme::UMIST>
                            (i, j, k, n, AMREX_D_DECL(uxface_fab, uyface_fab, uzface_fab),
                             vel_fab, dxinv, blend_factor);
                        Usrc_fab(i, j, k, n) = - gp_fab(i, j, k, n) + deferred_correction_term;
                    });
            }
        }
    }

    {
        BL_PROFILE("macro");

        for (int i = 0; i < 100; ++i) {
            const auto& dxinv = geom.InvCellSizeArray();
            const Real blend_factor = 0.5;
            AMREX_D_TERM(auto& uxface = velface[0];,
                         auto& uyface = velface[1];,
                         auto& uzface = velface[2];);

            MFItInfo mfi_info;
            if (Gpu::notInLaunchRegion()) { mfi_info.EnableTiling().SetDynamic(true); }
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(vel, mfi_info); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                const auto& vel_fab = vel.const_array(mfi);
                const auto& gp_fab = gp.const_array(mfi);
                const auto& Usrc_fab = Usrc.array(mfi);

                AMREX_D_TERM(const auto& uxface_fab = uxface.const_array(mfi);,
                             const auto& uyface_fab = uyface.const_array(mfi);,
                             const auto& uzface_fab = uzface.const_array(mfi););

                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, AMREX_SPACEDIM, i, j, k, n,
                {
                    amrex::Real deferred_correction_term = (deferred_correction<HOScheme::TVD, TVDScheme::UMIST>
                        (i, j, k, n, AMREX_D_DECL(uxface_fab, uyface_fab, uzface_fab),
                         vel_fab, dxinv, blend_factor));
                    Usrc_fab(i, j, k, n) = - gp_fab(i, j, k, n) + deferred_correction_term;
                });
            }
        }
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    amrex::Print() << "Running ParallelFor\n";

    testParallelFor();
    amrex::Finalize();
}
