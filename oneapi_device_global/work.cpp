
#include "global_vars.H"

void work ()
{
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int)
    {
        amrex::Real x = dg_x;
        auto xx = x + dg_x;
        AMREX_DEVICE_PRINTF("x = %g, y = %g, %g, %g, %g\n",
                            xx, dg_y[0], dg_y[1], dg_y[2], dg_y[3]);
    });
    amrex::Gpu::streamSynchronize();
}
