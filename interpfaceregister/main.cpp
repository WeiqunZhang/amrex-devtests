#include <AMReX.H>
#include <AMReX_InterpFaceRegister.H>
#include <AMReX_VisMF.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int n_cell = 128;
        {
            ParmParse pp;
            pp.query("n_cell", n_cell);
        }
        IntVect ref_ratio(2);

        Geometry cgeom, fgeom;
        {
            Box cdomain(IntVect(0), IntVect(n_cell-1));
            RealBox rb({AMREX_D_DECL(-1.,-1.,-1.)}, {AMREX_D_DECL(1.,1.,1.)});
            Array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(1,0,0)};
            cgeom = Geometry(cdomain, rb, CoordSys::cartesian, isperiodic);
            fgeom = Geometry(amrex::refine(cdomain,ref_ratio), rb, CoordSys::cartesian, isperiodic);
        }

        BoxArray cba, fba;
        {
            Box cdomain(IntVect(0), IntVect(n_cell-1));
            cba.define(cdomain);
            cba.maxSize(n_cell/4);

            int boxsize = n_cell/2;
            int blocksize = boxsize/2;

            BoxList bl;
            Box b(IntVect(0),IntVect(boxsize-1));
            for (int ishift = 0; ishift < 7; ++ishift) {
                bl.push_back(b);
                b.shift(IntVect(blocksize));
            }
            b = Box(IntVect(0),IntVect(boxsize-1));
            b.setSmall(0, n_cell*2-boxsize);
            b.setBig  (0, n_cell*2-1      );
            b.shift(1, blocksize);
            Box fdomain = amrex::refine(cdomain, ref_ratio);
            for (int ishift = 0; ishift <7; ++ishift) {
                bl.push_back(b & fdomain);
                b.shift(IntVect(AMREX_D_DECL(-blocksize,blocksize,blocksize)));
            }

            fba.define(std::move(bl));
            fba.removeOverlap();
        }

        const auto problo = cgeom.ProbLoArray();
        const auto cdx = cgeom.CellSizeArray();
        const auto fdx = fgeom.CellSizeArray();

        DistributionMapping cdm(cba);
        Array<MultiFab,AMREX_SPACEDIM> cmf;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            cmf[idim].define(amrex::convert(cba, IntVect::TheDimensionVector(idim)),
                             cdm, 1, 0);
            for (MFIter mfi(cmf[idim]); mfi.isValid(); ++mfi) {
                auto const& fab = cmf[idim].array(mfi);
                Box const& box = mfi.validbox();
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real x = problo[0] + Real(i)*cdx[0];
                    if (idim != 0) {
                        x += 0.5_rt * cdx[0];
                    }
                    Real y = problo[1] + Real(j)*cdx[1];
                    if (idim != 1) {
                        y += 0.5_rt * cdx[1];
                    }
                    constexpr Real pi = 3.141592653589793238462643383279502884197;
                    fab(i,j,k) = std::sin(pi*(x+0.25_rt)) * std::cos(pi*(y+0.25_rt));
                });
            }
        }

        DistributionMapping fdm(fba);
        Array<MultiFab,AMREX_SPACEDIM> fmf;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            fmf[idim].define(amrex::convert(fba, IntVect::TheDimensionVector(idim)),
                             fdm, 1, 0);
            for (MFIter mfi(fmf[idim]); mfi.isValid(); ++mfi) {
                auto const& fab = fmf[idim].array(mfi);
                Box const& box = mfi.validbox();
                ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real x = problo[0] + Real(i)*fdx[0];
                    if (idim != 0) {
                        x += 0.5_rt * fdx[0];
                    }
                    Real y = problo[1] + Real(j)*fdx[1];
                    if (idim != 1) {
                        y += 0.5_rt * fdx[1];
                    }
                    constexpr Real pi = 3.141592653589793238462643383279502884197;
                    fab(i,j,k) = std::sin(pi*(x+0.25_rt)) * std::cos(pi*(y+0.25_rt));
                });
            }
        }

        InterpFaceRegister ifr(fba, fdm, fgeom, ref_ratio);

        // Set coarse/fine boundary faces to zero
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            auto const& mlo_mf = ifr.mask(Orientation(idim,Orientation::low));
            auto const& mhi_mf = ifr.mask(Orientation(idim,Orientation::high));
            for (MFIter mfi(fmf[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.tilebox();
                Box const& vbx = mfi.validbox();
                int vlo = vbx.smallEnd(idim);
                int vhi = vbx.bigEnd(idim);
                auto const& fab = fmf[idim].array(mfi);
                auto const& mlo = mlo_mf.const_array(mfi);
                auto const& mhi = mhi_mf.const_array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    IntVect iv(AMREX_D_DECL(i,j,k));
                    if (iv[idim] == vlo && mlo(i,j,k)) { // lo-end crse/fine boundary
                        fab(i,j,k) = 2._rt;
                    } else if (iv[idim] == vhi && mhi(i,j,k)) { // hi-end crse/fine boundary
                        fab(i,j,k) = 2._rt;
                    }
                });
            }
        }

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            VisMF::Write(cmf[idim], "cmf-"+std::to_string(idim));
            VisMF::Write(fmf[idim], "a-fmf-"+std::to_string(idim));
        }

        // Interpolate from coarse to fine at coarse/fine boundary
        ifr.interp(amrex::GetArrOfPtrs(fmf), amrex::GetArrOfConstPtrs(cmf), 0, 1);        

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            VisMF::Write(fmf[idim], "b-fmf-"+std::to_string(idim));
        }
    }
    amrex::Finalize();
}
