#include <AMReX_EdgeFluxRegister.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int n_cell = 64;
        int max_grid_size = 8;
        Box domain(IntVect(0), IntVect(n_cell-1));
        RealBox rb({AMREX_D_DECL(0.0,0.0,0.0)}, {AMREX_D_DECL(1.0,1.0,1.0)});
        Array<int,AMREX_SPACEDIM> is_per{{AMREX_D_DECL(0,0,0)}};
        Geometry cgeom(domain, rb, 0, is_per);
        Geometry fgeom = amrex::refine(cgeom, 2);

        BoxArray cba(cgeom.Domain());
        cba.maxSize(max_grid_size);

        BoxArray fba;
        {
            Box fdomain = amrex::refine(domain,2);
            fdomain.grow(-n_cell/2);
            Box b2 = fdomain;
            b2.grow(-8);
            b2.shift(1,8);
            fba = amrex::boxComplement(fdomain,b2);
            fba.maxSize(max_grid_size);
        }

        DistributionMapping cdm(cba);
        DistributionMapping fdm(fba);

        Array<MultiFab,AMREX_SPACEDIM> B_crse, B_fine;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            B_crse[idim].define(amrex::convert(cba, IntVect::TheDimensionVector(idim)), cdm, 1, 0);
            B_fine[idim].define(amrex::convert(fba, IntVect::TheDimensionVector(idim)), fdm, 1, 0);
        }

#if (AMREX_SPACEDIM == 3)
        Array<MultiFab,3> E_crse, E_fine;
        for (int idim = 0; idim < 3; ++idim) {
            IntVect itype(1);
            itype[idim] = 0;
            E_crse[idim].define(amrex::convert(cba, itype), cdm, 1, 0);
            E_fine[idim].define(amrex::convert(fba, itype), fdm, 1, 0);
        }
#else
        MultiFab E_crse(amrex::convert(cba, IntVect::TheNodeVector()), cdm, 1, 0);
        MultiFab E_fine(amrex::convert(fba, IntVect::TheNodeVector()), fdm, 1, 0);
#endif

        EdgeFluxRegister efr(fba, cba, fdm, cdm, fgeom, cgeom);

        const Real dt_crse = cgeom.CellSize(0);
        const Real dt_fine = Real(0.5)*dt_crse;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            B_crse[idim].setVal(1.0);
            B_fine[idim].setVal(1.0);
        }

        efr.reset();

#if (AMREX_SPACEDIM == 3)
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            auto const& cma = E_crse[idim].arrays();
            auto const& cdx = cgeom.CellSizeArray();
            ParallelFor(E_crse[idim],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                GpuArray<Real,3> pos{i*cdx[0], j*cdx[1], k*cdx[2]};
                pos[idim] += Real(0.5)*cdx[idim];
                cma[bno](i,j,k) = std::sin(pos[0]) * std::cos(pos[1])
                    * std::cos(Real(2.)*pos[2]);
            });
            auto const& fma = E_fine[idim].arrays();
            auto const& fdx = fgeom.CellSizeArray();
            ParallelFor(E_fine[idim],
            [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                GpuArray<Real,3> pos{i*fdx[0], j*fdx[1], k*fdx[2]};
                pos[idim] += Real(0.5)*fdx[idim];
                fma[bno](i,j,k) = std::sin(pos[0]) * std::cos(pos[1])
                    * std::cos(Real(2.)*pos[2]);
            });
        }
        Gpu::synchronize();

        for (MFIter mfi(B_crse[0]); mfi.isValid(); ++mfi)
        {
            Box const& xbx = amrex::convert(mfi.validbox(),B_crse[0].ixType().ixType());
            Box const& ybx = amrex::convert(mfi.validbox(),B_crse[1].ixType().ixType());
            Box const& zbx = amrex::convert(mfi.validbox(),B_crse[2].ixType().ixType());
            auto const& Bxfab = B_crse[0].array(mfi);
            auto const& Byfab = B_crse[1].array(mfi);
            auto const& Bzfab = B_crse[2].array(mfi);
            auto const& Exfab = E_crse[0].const_array(mfi);
            auto const& Eyfab = E_crse[1].const_array(mfi);
            auto const& Ezfab = E_crse[2].const_array(mfi);
            Real dxi = cgeom.InvCellSize(0);
            Real dyi = cgeom.InvCellSize(1);
            Real dzi = cgeom.InvCellSize(2);
            ParallelFor(xbx, ybx, zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real ce = (Ezfab(i,j+1,k)-Ezfab(i,j,k))*dyi
                        -     (Eyfab(i,j,k+1)-Eyfab(i,j,k))*dzi;
                    Bxfab(i,j,k) -= dt_crse * ce;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real ce = (Exfab(i,j,k+1)-Exfab(i,j,k))*dzi
                        -     (Ezfab(i+1,j,k)-Ezfab(i,j,k))*dxi;
                    Byfab(i,j,k) -= dt_crse * ce;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real ce = (Eyfab(i+1,j,k)-Eyfab(i,j,k))*dxi
                        -     (Exfab(i,j+1,k)-Exfab(i,j,k))*dyi;
                    Bzfab(i,j,k) -= dt_crse * ce;
                });
            efr.CrseAdd(mfi, {&E_crse[0][mfi],
                              &E_crse[1][mfi],
                              &E_crse[2][mfi]}, dt_crse);
        }

        for (int cycle = 0; cycle < 2; ++cycle) {
            for (MFIter mfi(B_fine[0]); mfi.isValid(); ++mfi)
            {
                Box const& xbx = amrex::convert(mfi.validbox(),B_fine[0].ixType().ixType());
                Box const& ybx = amrex::convert(mfi.validbox(),B_fine[1].ixType().ixType());
                Box const& zbx = amrex::convert(mfi.validbox(),B_fine[2].ixType().ixType());
                auto const& Bxfab = B_fine[0].array(mfi);
                auto const& Byfab = B_fine[1].array(mfi);
                auto const& Bzfab = B_fine[2].array(mfi);
                auto const& Exfab = E_fine[0].const_array(mfi);
                auto const& Eyfab = E_fine[1].const_array(mfi);
                auto const& Ezfab = E_fine[2].const_array(mfi);
                Real dxi = fgeom.InvCellSize(0);
                Real dyi = fgeom.InvCellSize(1);
                Real dzi = fgeom.InvCellSize(2);
                ParallelFor(xbx, ybx, zbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real ce = (Ezfab(i,j+1,k)-Ezfab(i,j,k))*dyi
                            -     (Eyfab(i,j,k+1)-Eyfab(i,j,k))*dzi;
                        Bxfab(i,j,k) -= dt_fine * ce;
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real ce = (Exfab(i,j,k+1)-Exfab(i,j,k))*dzi
                            -     (Ezfab(i+1,j,k)-Ezfab(i,j,k))*dxi;
                        Byfab(i,j,k) -= dt_fine * ce;
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real ce = (Eyfab(i+1,j,k)-Eyfab(i,j,k))*dxi
                            -     (Exfab(i,j+1,k)-Exfab(i,j,k))*dyi;
                        Bzfab(i,j,k) -= dt_fine * ce;
                    });
                efr.FineAdd(mfi, {&E_fine[0][mfi],
                                  &E_fine[1][mfi],
                                  &E_fine[2][mfi]}, dt_fine);
            }
        }

#else

        auto const& cma = E_crse.arrays();
        auto const& cdx = cgeom.CellSizeArray();
        ParallelFor(E_crse,
        [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            Real pos0 = i*cdx[0] * 1.001;
            Real pos1 = j*cdx[1] * 1.001;
            cma[bno](i,j,k) = std::sin(pos0) * std::cos(Real(2.)*pos1);
        });
        auto const& fma = E_fine.arrays();
        auto const& fdx = fgeom.CellSizeArray();
        ParallelFor(E_fine,
        [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            Real pos0 = i*fdx[0];
            Real pos1 = j*fdx[1];
            fma[bno](i,j,k) = std::sin(pos0) * std::cos(Real(2.)*pos1);
        });
        Gpu::synchronize();

        for (MFIter mfi(B_crse[0]); mfi.isValid(); ++mfi)
        {
            Box const& xbx = amrex::convert(mfi.validbox(),B_crse[0].ixType().ixType());
            Box const& ybx = amrex::convert(mfi.validbox(),B_crse[1].ixType().ixType());
            auto const& Bxfab = B_crse[0].array(mfi);
            auto const& Byfab = B_crse[1].array(mfi);
            auto const& Ezfab = E_crse.const_array(mfi);
            Real dxi = cgeom.InvCellSize(0);
            Real dyi = cgeom.InvCellSize(1);
            ParallelFor(xbx, ybx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real ce = (Ezfab(i,j+1,k)-Ezfab(i,j,k))*dyi;
                    Bxfab(i,j,k) -= dt_crse * ce;
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real ce = -(Ezfab(i+1,j,k)-Ezfab(i,j,k))*dxi;
                    Byfab(i,j,k) -= dt_crse * ce;
                });
            efr.CrseAdd(mfi, E_crse[mfi], dt_crse);
        }

        for (int cycle = 0; cycle < 2; ++cycle) {
            for (MFIter mfi(B_fine[0]); mfi.isValid(); ++mfi)
            {
                Box const& xbx = amrex::convert(mfi.validbox(),B_fine[0].ixType().ixType());
                Box const& ybx = amrex::convert(mfi.validbox(),B_fine[1].ixType().ixType());
                auto const& Bxfab = B_fine[0].array(mfi);
                auto const& Byfab = B_fine[1].array(mfi);
                auto const& Ezfab = E_fine.const_array(mfi);
                Real dxi = fgeom.InvCellSize(0);
                Real dyi = fgeom.InvCellSize(1);
                ParallelFor(xbx, ybx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real ce = (Ezfab(i,j+1,k)-Ezfab(i,j,k))*dyi;
                        Bxfab(i,j,k) -= dt_fine * ce;
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real ce = -(Ezfab(i+1,j,k)-Ezfab(i,j,k))*dxi;
                        Byfab(i,j,k) -= dt_fine * ce;
                    });
                efr.FineAdd(mfi, E_fine[mfi], dt_fine);
            }
        }

#endif

        efr.Reflux(GetArrOfPtrs(B_crse));

        average_down_faces(GetArrOfConstPtrs(B_fine), GetArrOfPtrs(B_crse),
                           IntVect(2), cgeom);

        MultiFab cdiv(cba, cdm, 1, 0);
        MultiFab fdiv(fba, fdm, 1, 0);
        computeDivergence(cdiv, GetArrOfConstPtrs(B_crse), cgeom);
        computeDivergence(fdiv, GetArrOfConstPtrs(B_fine), fgeom);

        amrex::Print() << "div B on crse level: " << cdiv.min(0) << " "
                       << cdiv.max(0) << std::endl;
        amrex::Print() << "div B on fine level: " << fdiv.min(0) << " "
                       << fdiv.max(0) << std::endl;

        WriteMLMF("plot", {&cdiv, &fdiv}, {cgeom, fgeom});
    }
    amrex::Finalize();
}
