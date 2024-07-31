
#include "global_vars.H"

void work ()
{
    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int)
    {
        AMREX_DEVICE_PRINTF("x = %g, y = %g, %g, %g, %g\n",
                            dg_x.get(), dg_y[0], dg_y[1], dg_y[2], dg_y[3]);
    });
    amrex::Gpu::streamSynchronize();
}
