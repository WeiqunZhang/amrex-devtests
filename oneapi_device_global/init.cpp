
#include "global_vars.H"

void init ()
{
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int i)
    {
        dg_x = 1.1;
        for (int n = 0; n < 4; ++n) {
            dg_y[n] = amrex::Real(100 + n);
        }
    });

    amrex::Gpu::streamSynchronize();
}
