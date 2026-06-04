#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <Kokkos_Core.hpp>

using namespace amrex;

struct MinMaxSum {
    Real min_val, max_val, sum_val;
};

struct MinMaxSumReducer {
    using value_type = MinMaxSum;

    Real const* p = nullptr;

    KOKKOS_INLINE_FUNCTION void init (value_type& v) const {
        v.min_val = std::numeric_limits<Real>::max();
        v.max_val = std::numeric_limits<Real>::lowest();
        v.sum_val = 0;
    }

    KOKKOS_INLINE_FUNCTION void join (value_type& dst, value_type const& src) const {
        dst.min_val = std::min(dst.min_val, src.min_val);
        dst.max_val = std::max(dst.max_val, src.max_val);
        dst.sum_val += src.sum_val;
    }

    KOKKOS_INLINE_FUNCTION void operator() (std::size_t i, value_type& v) const {
        v.min_val = std::min(v.min_val, p[i]);
        v.max_val = std::max(v.max_val, p[i]);
        v.sum_val += std::abs(p[i]);
    }
};

static void test_amrex (Real const* p, std::size_t n, Real& rmin, Real& rmax, Real& rsum)
{
    Reducer<TypeList<ReduceOpMin,ReduceOpMax,ReduceOpSum>,
            TypeList<Real,Real,Real>> reducer{};
    reducer.eval(n, [=] AMREX_GPU_DEVICE (std::size_t i) -> GpuTuple<Real,Real,Real>
                 {
                     return {p[i], p[i], std::abs(p[i])};
                 });
    auto result = reducer.getResult();
    rmin += amrex::get<0>(result);
    rmax += amrex::get<1>(result);
    rsum += amrex::get<2>(result);
}

static void test_kokkos (Real const* p, std::size_t n, Real& rmin, Real& rmax, Real& rsum)
{
    MinMaxSum result;
    Kokkos::parallel_reduce("MinMaxSumReducer", n, MinMaxSumReducer(p), result);
    rmin += result.min_val;
    rmax += result.max_val;
    rsum += result.sum_val;
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    Kokkos::initialize(Kokkos::InitializationSettings().set_device_id(amrex::Gpu::Device::deviceId()));
    {
        int n = 256;
        ParmParse pp;
        pp.query("n", n);
        Gpu::DeviceVector<Real> dv(Long(n)*Long(n)*Long(n));
        FillRandom(dv.data(), dv.size());
        Real amin=0, amax=0, asum=0;
        Real kmin=0, kmax=0, ksum=0;
        double tamrex = 1.e10;
        double tkokkos = 1.e10;
        for (int count = 0; count < 10; ++count) {
            amrex::Gpu::synchronize();
            double t0 = amrex::second();

            test_amrex(dv.data(), dv.size(), amin, amax, asum);

            amrex::Gpu::synchronize();
            double t1 = amrex::second();

            test_kokkos(dv.data(), dv.size(), kmin, kmax, ksum);

            amrex::Gpu::synchronize();
            double t2 = amrex::second();

            tamrex = std::min(t1-t0, tamrex);
            tkokkos = std::min(t2-t1, tkokkos);
        }
        std::cout << "amrex  results: " << amin << " " << amax << " " << asum << "\n"
                  << "kokkos results: " << kmin << " " << kmax << " " << ksum << "\n\n";
        std::cout << "amrex  run time is " << tamrex << "\n"
                  << "kokkos run time is " << tkokkos << "\n";
    }
    Kokkos::finalize();
    amrex::Finalize();
}
