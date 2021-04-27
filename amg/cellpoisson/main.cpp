#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include <AMReX_AlgVector.H>
#include <AMReX_SpMatrix.H>

using namespace amrex;

void main_main ()
{
    int n_cell = 64;
    int max_grid_size = 16;

    Geometry geom;
    {
        RealBox rb({AMREX_D_DECL(0.,0.,0.), AMREX_D_DECL(1.,1.,1.)});
        std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,0,0)};
        Box domain(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n_cell-1,n_cell-1,n_cell-1)});
        geom.define(domain, rb, 0, isperiodic);
    }
    const auto prob_lo = geom.ProbLoArray();
    const auto dx      = geom.CellSizeArray();

    BoxArray ba(geom.Domain());
    ba.maxSize(max_grid_size);

    DistributionMapping dm(ba);

    MultiFab phi(ba, dm, 1, 1);
    MultiFab rhs(ba, dm, 1, 0);
    for (MFIter mfi(rhs); mfi.isValid(); ++mfi)
    {
        Box const& bx = mfi.tilebox();
        auto rhsfab = rhs.array(mfi);
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            constexpr amrex::Real tpi = 2.*3.1415926535897932;
            constexpr amrex::Real fpi = 4.*3.1415926535897932;
            constexpr amrex::Real fac = tpi*tpi*static_cast<amrex::Real>(AMREX_SPACEDIM);
#if (AMREX_SPACEDIM == 2)
            amrex::ignore_unused(k);
            amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
            amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
            rhsfab(i,j,k) = -fac * (std::sin(tpi*x) * std::sin(tpi*y))
                            -fac * (std::sin(fpi*x) * std::sin(fpi*y));
#else
            amrex::Real x = prob_lo[0] + dx[0] * (i + 0.5);
            amrex::Real y = prob_lo[1] + dx[1] * (j + 0.5);
            amrex::Real z = prob_lo[2] + dx[2] * (k + 0.5);
            rhsfab(i,j,k) = -fac * (std::sin(tpi*x) * std::sin(tpi*y) * std::sin(tpi*z))
                            -fac * (std::sin(fpi*x) * std::sin(fpi*y) * std::sin(fpi*z));
#endif
        });
    }
    phi.setVal(0.0);


}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
