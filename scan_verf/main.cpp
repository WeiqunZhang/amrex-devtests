#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <AMReX_Scan.H>

#include <limits>
#include <numeric>

using namespace amrex;

template <typename T>
int test_scan (unsigned int N, bool debug)
{
    bool ret = true;

    Gpu::DeviceVector<T> dv(N);
    Gpu::DeviceVector<T> dv_out(N);
    T* dp = dv.data();
    T* dpo = dv_out.data();
    ParallelForRNG(N, [=] AMREX_GPU_DEVICE (unsigned int i, RandomEngine const& engine)
    {
        dp[i] = static_cast<T>((Random(engine)-0.5)*100.);
    });
    Gpu::synchronize();

    if (debug) {
        amrex::Print() << "    Finished init" << std::endl;
    }

    Gpu::DeviceVector<T> bm(N);
    T* dpbm = bm.data();
#if defined(AMREX_USE_DPCPP) && !defined(AMREX_USE_ONEDPL)
    Gpu::PinnedVector<T> hbm(N);
    Gpu::dtoh_memcpy(hbm.data(), dp, sizeof(T)*N);
    Gpu::synchronize();
    std::inclusive_scan(hbm.begin(), hbm.end(), hbm.begin(), std::plus<T>());
    T bm_sum = hbm.back();
    Gpu::htod_memcpy(bm.data(), hbm.data(), sizeof(T)*N);
    Gpu::synchronize();
#else
    Gpu::inclusive_scan(dv.begin(), dv.end(), bm.begin());
    T bm_sum;
    Gpu::dtoh_memcpy(&bm_sum, dpbm+N-1, sizeof(T));
    Gpu::synchronize();
#endif

    if (debug) {
        amrex::Print() << "    Finished inclusive_scan for "
                       << (sizeof(T) == 8 ? "long" : "int") << std::endl;
    }

    { // inclusive scan
        bool ret_inc = true;
        T sum =  Scan::PrefixSum<T>(N, [=] AMREX_GPU_DEVICE (unsigned i) { return dp[i]; },
                                    [=] AMREX_GPU_DEVICE (unsigned i, T ps) { return dpo[i] = ps; },
                                    Scan::Type::inclusive);
        ret_inc = ret_inc && (sum == bm_sum);
        if (debug && !ret_inc) {
            amrex::Print() << "    Inclusive sum failed with wrong total sum "
                           << sum << ".  Should be " << bm_sum << std::endl;
        }
        T error = Reduce::Sum<T>(N, [=] AMREX_GPU_DEVICE (unsigned i) -> T {
                return amrex::Math::abs(dpo[i] - dpbm[i]);
            });
        ret_inc = ret_inc && (error == T{0});
        if (debug) {
            if (ret_inc) {
                amrex::Print() << "    Inclusive sum passed" << std::endl;;
            } else {
                amrex::Print() << "    Inclusive sum failed" << std::endl;;
            }
        }
        ret = ret && ret_inc;
    }

    { // exclusive scan
        bool ret_exc = true;
        T sum =  Scan::PrefixSum<T>(N, [=] AMREX_GPU_DEVICE (unsigned i) { return dp[i]; },
                                    [=] AMREX_GPU_DEVICE (unsigned i, T ps) { return dpo[i] = ps; },
                                    Scan::Type::exclusive);
        ret_exc = ret_exc && (sum == bm_sum);
        if (debug && !ret_exc) {
            amrex::Print() << "    Exclusive sum failed with wrong total sum "
                           << sum << ".  Should be " << bm_sum << std::endl;
        }
        T error = Reduce::Sum<T>(N, [=] AMREX_GPU_DEVICE (unsigned i) -> T {
                if (i == 0) {
                    return T{0};
                } else {
                    return amrex::Math::abs(dpo[i] - dpbm[i-1]);
                }
            });
        ret_exc = ret_exc && (error == T{0});
        if (debug) {
            if (ret_exc) {
                amrex::Print() << "    Exclusive sum passed" << std::endl;
            } else {
                amrex::Print() << "    Exclusive sum failed" << std::endl;
            }
        }
        ret = ret && ret_exc;
    }

    return ret;
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Long ntests = std::numeric_limits<Long>::max();
        unsigned int max_size = 200'000'000u;
        unsigned int min_size = 1u;
        double max_run_seconds, report_int_seconds;
        unsigned int single_test_size = 0;
        bool debug = false;
        bool test_long = true;
        {
            ParmParse pp;

            pp.query("ntests", ntests);

            double max_run_minutes = 3.;
            pp.query("max_run_minutes", max_run_minutes);
            max_run_seconds = max_run_minutes * 60.;

            double report_int_minutes = 1;
            pp.query("report_int_minutes", report_int_minutes);
            report_int_seconds = report_int_minutes * 60.;

            int tmp = -1;
            pp.query("max_size", tmp);
            if (tmp > 0) { max_size = tmp; }

            tmp = -1;
            pp.query("min_size", tmp);
            if (tmp > 0) { min_size = tmp; }

            tmp = -1;
            pp.query("single_test_size", tmp);
            if (tmp > 0) { single_test_size = tmp; }

            pp.query("debug", debug);
            pp.query("test_long", test_long);
        }
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(single_test_size > 0 || max_size >= min_size,
                                         "min_size must be <= max_size");
        double t_begin = amrex::second();
        double t_end = t_begin + max_run_seconds;
        double t_report = t_begin + report_int_seconds;
        Long ntot = 0, npass_int = 0, npass_long = 0;
        if (single_test_size > 0) { ntests = 1; }
        for (Long k = 0; k < ntests; ++k) {
            if (amrex::second() > t_end) { break; }
            unsigned int N;
            if (single_test_size > 0) {
                N = single_test_size;
            } else {
                N = amrex::Random_int(max_size-min_size+1) + min_size;
            }
            if (debug) {
                amrex::Print() << "# " << k << ": N = " << N << std::endl;
            }
            ++ntot;
            npass_int += test_scan<int>(N, debug);
            if (test_long) {
                npass_long += test_scan<Long>(N, debug);
            } else {
                ++npass_long;
            }
            if (amrex::second() > t_report) {
                t_report += report_int_seconds;
                amrex::Print() << "After running " << ntot << " tests in "
                               << static_cast<int>((amrex::second()-t_begin)/6.)*0.1
                               << " minutes, ";
                Long nfail_int = ntot - npass_int;
                Long nfail_long = ntot - npass_long;
                if (nfail_int == 0 && nfail_long == 0) {
                    amrex::Print() << "no tests failed.";
                }
                if (nfail_int > 0) {
                    amrex::Print() << nfail_int << " int tests failed. ";
                }
                if (nfail_long > 0) {
                    amrex::Print() << nfail_long << " long int tests failed.";
                }
                amrex::Print() << std::endl;
            }
        }
    }
    amrex::Finalize();
}
