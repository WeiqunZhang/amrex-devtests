#include <AMReX.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int ncells = 256;
        int nx = ncells - 2;
        int ny = ncells - 1;
        int nz = ncells + 2;
        Box box(IntVect(0), IntVect(AMREX_D_DECL(nx-1,ny-1,nz-1)));
#if (AMREX_SPACEDIM == 3)
        Box cbox(IntVect(0), IntVect(nz-1,ny-1,nx-1));
#else
        Box cbox(IntVect(0), IntVect(ny-1,nx-1));
#endif
        FArrayBox fab1(cbox);
        FArrayBox fab2(box);
        auto const& a1 = fab1.array();
        auto const& a2 = fab2.array();

        auto value = [=] AMREX_GPU_DEVICE (int i, int j, int k) -> Real
        {
            return i + j*nx + k*nx*ny;
        };

        amrex::ParallelFor(cbox, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
#if (AMREX_SPACEDIM == 3)
            a1(i,j,k) = value(k,j,i);
#else
            a1(i,j,k) = value(j,i,k);
#endif
        });

        amrex::transposeCtoF(fab1.dataPtr(), fab2.dataPtr(),
                             AMREX_D_DECL(box.length(0),
                                          box.length(1),
                                          box.length(2)));

        Gpu::streamSynchronize();
        auto t0 = amrex::second();
        for (int i = 0; i < 10; ++i) {
            amrex::transposeCtoF(fab1.dataPtr(), fab2.dataPtr(),
                                 AMREX_D_DECL(box.length(0),
                                              box.length(1),
                                              box.length(2)));
        }
        Gpu::streamSynchronize();
        auto t1 = amrex::second();

        amrex::Print() << "  Run Time: " << t1-t0 << "\n";

        amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            a2(i,j,k) -= value(i,j,k);
        });

        auto error = fab2.template norm<RunOn::Device>(0);
        AMREX_ALWAYS_ASSERT(error == 0);
    }
    amrex::Finalize();
}
