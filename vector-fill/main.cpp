#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuComplex.H>
#include <complex>

using namespace amrex;

struct Foo {
    double x[5];
    int i;
    bool operator!= (Foo const& rhs) {
        for (int j = 0; j < 5; ++j) {
            if (this->x[j] != rhs.x[j]) return true;
        }
        return this->i != rhs.i;
    }
};

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        const int N = 2550000;

        static_assert(std::is_trivially_copyable<Foo>());
        //static_assert(std::is_trivially_copyable<std::pair<Real,int>>());
        static_assert(std::is_trivially_copyable<std::complex<Real>>());
        static_assert(std::is_trivially_copyable<amrex::GpuComplex<Real>>());
        static_assert(std::is_trivially_copyable<std::array<Real,10>>());
        static_assert(std::is_trivially_copyable<amrex::GpuArray<Real,10>>());
        static_assert(std::is_trivially_copyable<amrex::Array1D<Real,-3,10>>());
        static_assert(std::is_trivially_copyable<amrex::Array2D<int,4,8,-3,10>>());

        Gpu::DeviceVector<Foo> dv(N);
        Gpu::PinnedVector<Foo> hv(N);

        double t0, t1;
        for (int it = 0; it < 3; ++it) {
            t0 = amrex::second();
            Gpu::fillAsync(dv.begin(), dv.end(),
                           [=] AMREX_GPU_DEVICE (Foo& a, Long i) {
                               a.x[0] = 1.1 + i;
                               a.x[1] = 2.2 + i;
                               a.x[2] = 3.3 + i;
                               a.x[3] = 4.4 + i;
                               a.x[4] = 5.5 + i;
                               a.i = 100 + i;
                           });
            Gpu::streamSynchronize();
            t1 = amrex::second();

            Gpu::copyAsync(Gpu::deviceToHost, dv.begin(), dv.end(), hv.begin());
            Gpu::streamSynchronize();
            for (Long i = 0; i < N; ++i) {
                Foo a;
                a.x[0] = 1.1 + i;
                a.x[1] = 2.2 + i;
                a.x[2] = 3.3 + i;
                a.x[3] = 4.4 + i;
                a.x[4] = 5.5 + i;
                a.i = 100 + i;
                if (a != hv[i]) {
                    amrex::Abort("Gpu::fillAsync failed");
                }
            }
        }

        double t2, t3;
        for (int it = 0; it < 3; ++it) {
            t2 = amrex::second();
            auto p = dv.data();
            amrex::ParallelFor(N, [=] AMREX_GPU_DEVICE (Long i) noexcept
            {
                p[i].x[0] = 1.1 + i;
                p[i].x[1] = 2.2 + i;
                p[i].x[2] = 3.3 + i;
                p[i].x[3] = 4.4 + i;
                p[i].x[4] = 5.5 + i;
                p[i].i = 100 + i;
            });
            Gpu::streamSynchronize();
            t3 = amrex::second();
        }

        amrex::Print() << "  Gpu::fillAsync time is " << t1-t0 << "\n"
                       << "  ParallelFor    time is " << t3-t2 << "\n";
    }
    amrex::Finalize();
}
