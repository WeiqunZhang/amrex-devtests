#include <work.H>
#include <par.H>
#include <AMReX_Gpu.H>

void work ()
{
    amrex::Gpu::PinnedVector<int> pv(10);
    int* ppv = pv.data();

    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int)
    {
        ppv[0] = d_a;
        ppv[1] = d_b[0];
        ppv[2] = d_b[1];
        ppv[3] = d_b[2];
        ppv[4] = d_b[3];
        ppv[5] = m_a;
        ppv[6] = m_b[0];
        ppv[7] = m_b[1];
        ppv[8] = m_b[2];
        ppv[9] = m_b[3];
    });

    amrex::Gpu::synchronize();

    amrex::Print() << pv[0] << ", " << pv[1] << ", " << pv[2] << ", " << pv[3] << ", " << pv[4] << ", "
                   << pv[5] << ", " << pv[6] << ", " << pv[7] << ", " << pv[8] << ", " << pv[9] << '\n';
}
