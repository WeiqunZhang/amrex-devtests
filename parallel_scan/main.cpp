#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
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
        dp[i] = static_cast<T>((Random()-0.5)*100.);
    });

    Gpu::PinnedVector<T> hv(N);
    Gpu::dtoh_memcpy_async(hv.data(), dv.data(), sizeof(T)*N);

    Gpu::PinnedVector<T> hv_out(N);
    Gpu::PinnedVector<T> dv_out_pinned(N);

    { // inclusive scan
        bool ret_inc = true;
        T sum =  Scan::PrefixSum<T>(N, [=] AMREX_GPU_DEVICE (unsigned i) { return dp[i]; },
                                    [=] AMREX_GPU_DEVICE (unsigned i, T ps) { return dpo[i] = ps; },
                                    Scan::Type::inclusive);
        Gpu::dtoh_memcpy_async(dv_out_pinned.data(), dv_out.data(), sizeof(T)*N);
        Gpu::synchronize();
        std::partial_sum(hv.begin(), hv.end(), hv_out.begin());
        ret_inc = ret_inc && (sum == hv_out[N-1]);
        if (debug && !ret_inc) {
            amrex::Print() << "    Inclusive sum failed with wrong total sum "
                           << sum << ".  Should be " << hv_out[N-1] << std::endl;
        }
        for (unsigned i = 0; i < N; ++i) {
            ret_inc = ret_inc && (dv_out_pinned[i] == hv_out[i]);
        }
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
        Gpu::dtoh_memcpy_async(dv_out_pinned.data(), dv_out.data(), sizeof(T)*N);
        Gpu::synchronize();
        ret_exc = ret_exc && (sum == hv_out[N-1]);
        if (debug && !ret_exc) {
            amrex::Print() << "    Exclusive sum failed with wrong total sum "
                           << sum << ".  Should be " << hv_out[N-1] << std::endl;
        }
        ret_exc = ret_exc && (dv_out_pinned[0] == T{0});
        for (unsigned i = 0; i < N-1; ++i) {
            ret_exc = ret_exc && (dv_out_pinned[i+1] == hv_out[i]);
        }
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
        double max_run_seconds, report_int_seconds;
        unsigned int single_test_size = 0;
        bool debug = false;
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
            pp.query("single_test_size", tmp);
            if (tmp > 0) { single_test_size = tmp; }

            pp.query("debug", debug);
        }
        double t_begin = amrex::second();
        double t_end = t_begin + max_run_seconds;
        double t_report = t_begin + report_int_seconds;
        Long ntot = 0, npass_int = 0, npass_long = 0;
        if (single_test_size > 0) { ntests = 1; }
        for (Long k = 0; k < ntests; ++k) {
            if (amrex::second() > t_end) { break; }
            unsigned int N = (single_test_size > 0)
                ? single_test_size : amrex::Random_int(max_size-1) + 1;
            if (debug) {
                amrex::Print() << "# " << k << ": N = " << N << std::endl;
            }
            ++ntot;
            npass_int += test_scan<int>(N, debug);
            npass_long += test_scan<Long>(N, debug);
            if (amrex::second() > t_report) {
                t_report += report_int_seconds;
                amrex::Print() << "After running " << ntot << " tests in "
                               << static_cast<int>((amrex::second()-t_begin)/60.)
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
