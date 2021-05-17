#include <work.H>
#include <par.H>
#include <AMReX_Gpu.H>

void work ()
{
    amrex::Gpu::PinnedVector<int> pv(4);
    int* ppv = pv.data();

    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int)
    {
        ppv[0] = d_a;
        ppv[1] = d_b[0];
        ppv[2] = d_b[1];
        ppv[3] = d_b[2];
    });

    amrex::Gpu::synchronize();

    amrex::Print() << pv[0] << ", " << pv[1] << ", " << pv[2] << ", " << pv[3] << std::endl;
}
