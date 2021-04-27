#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_Print.H>

#include <limits>

using namespace amrex;

void stream_init (Gpu::DeviceVector<Real>& av, Gpu::DeviceVector<Real>& bv,
                  Gpu::DeviceVector<Real>& cv)
{
    const amrex::Long N = av.size();
    Real* a = av.data();
    Real* b = bv.data();
    Real* c = cv.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    });
}

void stream_copy (Gpu::DeviceVector<Real> const& av, Gpu::DeviceVector<Real>& cv)
{
    const amrex::Long N = cv.size();
    Real const* a = av.data();
    Real* c = cv.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        c[i] = a[i];
    });
}

void stream_scale (Gpu::DeviceVector<Real>& bv, Gpu::DeviceVector<Real> const& cv, Real scalar)
{
    const amrex::Long N = bv.size();
    Real * b = bv.data();
    Real const* c = cv.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        b[i] = scalar*c[i];
    });
}

void stream_add (Gpu::DeviceVector<Real> const& av, Gpu::DeviceVector<Real> const& bv,
                 Gpu::DeviceVector<Real>& cv)
{
    const amrex::Long N = av.size();
    Real const * a = av.data();
    Real const* b = bv.data();
    Real* c = cv.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        c[i] = a[i] + b[i];
    });
}

void stream_triad (Gpu::DeviceVector<Real>& av, Gpu::DeviceVector<Real> const& bv,
                   Gpu::DeviceVector<Real> const& cv, Real scalar)
{
    const amrex::Long N = av.size();
    Real * a = av.data();
    Real const* b = bv.data();
    Real const* c = cv.data();
    amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (amrex::Long i)
    {
        a[i] = b[i] + scalar*c[i];
    });
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    amrex::SetVerbose(0);
    {
        Gpu::DeviceVector<Real> av(STREAM_ARRAY_SIZE);
        Gpu::DeviceVector<Real> bv(STREAM_ARRAY_SIZE);
        Gpu::DeviceVector<Real> cv(STREAM_ARRAY_SIZE);
        Real scalar = 3.0;

        stream_init(av,bv,cv);
        Gpu::synchronize();

        // warm up
        stream_copy(av,cv);
        stream_scale(bv,cv,scalar);
        stream_add(av,bv,cv);
        stream_triad(av,bv,cv,scalar);
        Gpu::synchronize();

        stream_init(av,bv,cv);
        Gpu::synchronize();

        Array2D<double,0,3,0,NTIMES-1,Order::C> times;
        for (int it = 0; it < NTIMES; ++it) {
            Real t = amrex::second();
            stream_copy(av, cv);
            Gpu::synchronize();
            times(0,it) = amrex::second() - t;

            t = amrex::second();
            stream_scale(bv, cv, scalar);
            Gpu::synchronize();
            times(1,it) = amrex::second() - t;

            t = amrex::second();
            stream_add(av, bv, cv);
            Gpu::synchronize();
            times(2,it) = amrex::second() - t;

            t = amrex::second();
            stream_triad(av, bv, cv, scalar);
            Gpu::synchronize();
            times(3,it) = amrex::second() - t;
        }

        Array<double,4> avgtime, mintime, maxtime;
        for (int j = 0; j < 4; ++j) {
            avgtime[j] = 0.0;
            mintime[j] = std::numeric_limits<Real>::max();
            maxtime[j] = std::numeric_limits<Real>::lowest();
            for (int it = 0; it < NTIMES; ++it) {
                avgtime[j] += times(j,it);
                mintime[j] = amrex::min(mintime[j], times(j,it));
                maxtime[j] = amrex::max(maxtime[j], times(j,it));
            }
            avgtime[j] /= double(NTIMES);
        }

        double bytes[4] = {
            2 * sizeof(Real) * STREAM_ARRAY_SIZE,
            2 * sizeof(Real) * STREAM_ARRAY_SIZE,
            3 * sizeof(Real) * STREAM_ARRAY_SIZE,
            3 * sizeof(Real) * STREAM_ARRAY_SIZE
        };

        char label[4][12] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

        std::printf("Function    Best Rate MB/s  Avg time     Min time     Max time\n");
        for (int j = 0; j < 4; ++j) {
            printf("%s%12.1f  %11.6f  %11.6f  %11.6f\n", label[j],
                   1.0e-06 * bytes[j]/mintime[j], avgtime[j], mintime[j], maxtime[j]);
        }
    }
    amrex::Finalize();
}
