#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_Print.H>

#include <AMReX_AlgVector.H>
#include <AMReX_BiCGSTAB.H>
#include <AMReX_RowIndex.H>
#include <AMReX_SpMatrix.H>

using namespace amrex;

void solve (Geometry const& geom, MultiFab& phi, MultiFab const& rhs)
{
    // Create cell id and partition
    RowIndex rowidx(rhs.boxArray(), rhs.DistributionMap(), geom, IntVect(1));
    FabArray<BaseFab<Long> > const& cell_id = rowidx.index();
    AlgPartition const& partition = rowidx.partition();

    // Create AlgVectors for phi and rhs
    AlgVector<Real> phivec(partition);
    AlgVector<Real> rhsvec(partition);
    phivec.copyFrom(phi);
    rhsvec.copyFrom(rhs);

    // Create SpMatrix for stencil
    constexpr int stencil_size = 2*AMREX_SPACEDIM+1;
    SpMatrix<Real> A(partition, stencil_size);

    const auto dxinv = geom.InvCellSizeArray();
    const GpuArray<Real,AMREX_SPACEDIM> fac{AMREX_D_DECL(dxinv[0]*dxinv[0],
                                                         dxinv[1]*dxinv[1],
                                                         dxinv[2]*dxinv[2])};

    // Set up matrix
    for (MFIter mfi(phi); mfi.isValid(); ++mfi) {
        Box const& vbx = mfi.validbox();
        auto const& cid = cell_id.const_array(mfi);

        Long row_begin = rowidx.range(mfi).first;

        Real* mat = A.data() + row_begin*stencil_size;
        AlgInt* col = A.columnIndex() + row_begin*stencil_size;

        BaseFab<GpuArray<Real,stencil_size> > tmpmatfab
            (vbx, 1, (GpuArray<Real,stencil_size>*)mat);
        amrex::fill(tmpmatfab,
        [=] AMREX_GPU_HOST_DEVICE (GpuArray<Real,stencil_size>& sten, int i, int j, int k)
        {
#if (AMREX_SPACEDIM == 2)
            sten[0] = Real(-2.) * (fac[0] + fac[1]);
            if (cid(i-1,j,k) >= 0) {
                sten[1] = fac[0];
            } else { // Homogeneous Dirichlet boundary
                sten[1] = Real(0.);
            }
            if (cid(i+1,j,k) >= 0) {
                sten[2] = fac[0];
            } else { // Homogeneous Dirichlet boundary
                sten[2] = Real(0.);
            }
            if (cid(i,j-1,k) >= 0) {
                sten[3] = fac[0];
            } else { // Homogeneous Dirichlet boundary
                sten[3] = Real(0.);
            }
            if (cid(i,j+1,k) >= 0) {
                sten[4] = fac[0];
            } else { // Homogeneous Dirichlet boundary
                sten[4] = Real(0.);
            }
#else
            static_assert(false, "3d todo");
#endif
        });

        BaseFab<GpuArray<AlgInt,stencil_size> > tmpcolfab
            (vbx, 1, (GpuArray<AlgInt,stencil_size>*)col);
        amrex::fill(tmpcolfab,
        [=] AMREX_GPU_HOST_DEVICE (GpuArray<AlgInt,stencil_size>& sten, int i, int j, int k)
        {
#if (AMREX_SPACEDIM == 2)
            sten[0] = cid(i,j,k);
            sten[1] = cid(i-1,j,k);
            sten[2] = cid(i+1,j,k);
            sten[3] = cid(i,j-1,k);
            sten[4] = cid(i,j+1,k);
#else
            static_assert(false, "3d todo");
#endif
        });
    }

    phivec.printToFile("phivec");
    rhsvec.printToFile("rhsvec");

    A.assemble();

    A.printToFile("A");

    bicgstab_solve(phivec, A, rhsvec, 1.e-4, 0.0);

    // Copy data back
    phivec.copyTo(phi);

    VisMF::Write(rhs, "rhs");
    VisMF::Write(phi, "phi");
}

void main_main ()
{
    int n_cell = 4;
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

    solve(geom, phi, rhs);
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    main_main();
    amrex::Finalize();
}
