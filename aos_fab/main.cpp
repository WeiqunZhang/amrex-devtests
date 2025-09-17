#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_MultiFab.H>
#include <AMReX_TableData.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncells = 64;
        BoxArray ba(Box(IntVect(0), IntVect(ncells-1)));
        ba.maxSize(32);
        DistributionMapping dm(ba);

        constexpr int ncomp = 4;
        int nghost = 2;
        FabArray<BaseFab<GpuArray<Real,ncomp>>> fa(ba,dm,1,nghost);

        auto value = [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) -> Real
        {
            if (i < 0) {
                i += ncells;
            } else if (i >= ncells) {
                i -= ncells;
            }
            if (j < 0) {
                j += ncells;
            } else if (j >= ncells) {
                j -= ncells;
            }
            if (k < 0) {
                k += ncells;
            } else if (k >= ncells) {
                k -= ncells;
            }
            return n + i*ncomp + j*ncomp*ncells + k*ncomp*ncells*ncells;
        };

        for (MFIter mfi(fa); mfi.isValid(); ++mfi) {
            auto const& a = fa.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                for (int n = 0; n < ncomp; ++n) {
                    a(i,j,k)[n] = value(i,j,k,n);
                }
            });
        }

        fa.FillBoundary(Periodicity(IntVect(ncells)));

        // We could use MultiFab/FArrayBox/BaseFab to store AoS data. But
        // almost all of their member functions will not work. We can still
        // access them with the help of Table4D or PolymorphicArray4.
        int ncomp_runtime = ncomp;
        MultiFab mf(ba,dm,ncomp_runtime,nghost);
        for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
#if defined(AMREX_USE_GPU)
            Gpu::dtod_memcpy_async
#else
            std::memcpy
#endif
                (mf[mfi].dataPtr(), fa[mfi].dataPtr(), mf[mfi].nBytes());

            Box const& bx = mf[mfi].box(); // must use the entire box.
            Table4D<Real> t(mf[mfi].dataPtr(),
                            {0,     bx.smallEnd(0), bx.smallEnd(1), bx.smallEnd(2)},
                            {ncomp, bx.bigEnd(0)+1, bx.bigEnd(1)+1, bx.bigEnd(2)+1});

            // We can access FabArray<BaseFab<GpuArray<>>> in two ways
            Array4<GpuArray<Real,ncomp>> const& a1 = fa[mfi].array();
            Table4D<Real> a2(reinterpret_cast<Real*>(fa[mfi].dataPtr()),
                             {0,     bx.smallEnd(0), bx.smallEnd(1), bx.smallEnd(2)},
                             {ncomp, bx.bigEnd(0)+1, bx.bigEnd(1)+1, bx.bigEnd(2)+1});

            auto pa = makePolymorphic(a1);

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                for (int n = 0; n < ncomp; ++n) {
                    AMREX_ALWAYS_ASSERT(t(n,i,j,k) == value(i,j,k,n));
                    AMREX_ALWAYS_ASSERT(t(n,i,j,k) == a1(i,j,k)[n]);
                    AMREX_ALWAYS_ASSERT(t(n,i,j,k) == a2(n,i,j,k));
                    AMREX_ALWAYS_ASSERT(t(n,i,j,k) == pa(i,j,k,n));
                }
            });
        }
    }
    amrex::Finalize();
}
