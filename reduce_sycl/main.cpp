#include <AMReX.H>
#include <AMReX_Gpu.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>

using namespace amrex;

template <typename T>
void test_reduce_sum (int npts)
{
    Vector<double> twall;
    Vector<T> result;
    Vector<std::string> desc;

    amrex::Gpu::DeviceVector<T> v(npts);
    T* p = v.data();
    amrex::ParallelForRNG(npts,
    [=] AMREX_GPU_DEVICE (int i, RandomEngine const& engine)
    {
        p[i] = static_cast<T>(amrex::Random(engine) + 0.5_rt);
    });
    Gpu::streamSynchronize();

    constexpr int ntests = 5;

    double ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n)
    {
        double t0 = amrex::second();
        auto r = Reduce::Sum(npts, p);
        double t1 = amrex::second();
        if (n == 0) {
            result.push_back(r);
            desc.push_back("    amrex  sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#if defined(AMREX_USE_CUB)

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();
        auto hsum = (T*)The_Pinned_Arena()->alloc(sizeof(T));
        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, hsum, npts);
        // Allocate temporary storage
        d_temp_storage = (void*)The_Arena()->alloc(temp_storage_bytes);
        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, p, hsum, npts);
        Gpu::streamSynchronize();
        The_Arena()->free(d_temp_storage);
        The_Pinned_Arena()->free(hsum);
        double t1 = amrex::second();
        if (n == 0) {
            result.push_back(*hsum);
            desc.push_back("    cub    sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#elif defined(AMREX_USE_HIP)

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();
        auto hsum = (T*)The_Pinned_Arena()->alloc(sizeof(T));
        void * d_temp_storage = nullptr;
        size_t temporary_storage_bytes = 0;
        rocprim::reduce(d_temp_storage, temporary_storage_bytes,
                        p, hsum, npts, rocprim::plus<T>());
        d_temp_storage = The_Arena()->alloc(temporary_storage_bytes);
        rocprim::reduce(d_temp_storage, temporary_storage_bytes,
                        p, hsum, npts, rocprim::plus<T>());
        Gpu::streamSynchronize();
        The_Arena()->free(d_temp_storage);
        The_Pinned_Arena()->free(hsum);
        double t1 = amrex::second();
        if (n == 0) {
            result.push_back(*hsum);
            desc.push_back("    hip    sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#elif defined(AMREX_USE_DPCPP)

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();

        T sumResult = 0;
        sycl::buffer<T> sumBuf { &sumResult, 1 };

        Gpu::Device::streamQueue().submit([&] (sycl::handler& cgh)
        {
            auto sumReduction = sycl::reduction(sumBuf, cgh, sycl::plus<>());

            cgh.parallel_for(sycl::range<1>{static_cast<std::size_t>(npts)}, sumReduction,
            [=] (sycl::id<1> idx, auto& sum)
            {
                sum += p[idx];
            });
        });
        sumResult = sumBuf.get_host_access()[0];

        double t1 = amrex::second();

        if (n == 0) {
            result.push_back(sumResult);
            desc.push_back("    sycl   sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();

        auto& gdev = amrex::Gpu::Device::syclDevice();
        auto& q    = amrex::Gpu::Device::streamQueue();

        int num_groups = gdev.get_info<sycl::info::device::max_compute_units>();
        int wgsize     = gdev.get_info<sycl::info::device::max_work_group_size>();

        auto d_sum = sycl::aligned_alloc_shared<T>(1024, num_groups, q);
        q.submit([&](sycl::handler& cgh)
        {
            auto wg_sum = sycl::accessor<T, 1, sycl::access::mode::read_write,
                                         sycl::access::target::local>
                (sycl::range<1>(wgsize), cgh);
            size_t N = npts;
            cgh.parallel_for(sycl::nd_range<1>(num_groups*wgsize, wgsize),
            [=] (sycl::nd_item<1> item)
            {
                size_t i  = item.get_global_id(0);
                size_t li = item.get_local_id(0);
                size_t global_size = item.get_global_range()[0];

                wg_sum[li] = 0;
                for (; i < N; i += global_size) {
                    wg_sum[li] += p[i];
                }

                size_t local_size = item.get_local_range()[0];

                for (int offset = local_size / 2; offset > 0; offset /= 2)
                {
                    item.barrier(sycl::access::fence_space::local_space);
                    if (li < offset) {
                        wg_sum[li] += wg_sum[li + offset];
                    }
                }

                if (li == 0) {
                    d_sum[item.get_group(0)] = wg_sum[0];
                }
            });
        });
        q.wait_and_throw();

        T sum = 0;
#if 0
        for (int i = 0; i < num_groups; ++i) {
            sum += d_sum[i]; // This will segfault on ATS.
        }
#endif
        sycl::free(d_sum, q);

        double t1 = amrex::second();

        if (n == 0) {
            result.push_back(sum);
            desc.push_back("    SLM    sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#if defined(AMREX_USE_ONEDPL)

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();

        auto policy = dpl::execution::make_device_policy(Gpu::Device::streamQueue());
        auto sumResult = std::reduce(policy, p, p+npts);

        double t1 = amrex::second();

        if (n == 0) {
            result.push_back(sumResult);
            desc.push_back("    onedpl sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#endif

    ttmp = 1.e6;
    for (int n = 0; n < ntests; ++n) {
        double t0 = amrex::second();

        // amrex reduction is reproduced here.
        // It seems that using amrex::The_Arena helps a lot.

#if 0
#define MY_ALLOC_DEVICE(x) sycl::malloc_device(x, dev, ctx)
#define MY_ALLOC_HOST(x)   sycl::malloc_host(x, ctx)
#define MY_FREE_DEVICE(x)  sycl::free(x, ctx);
#define MY_FREE_HOST(x)    sycl::free(x, ctx);
#else
#define MY_ALLOC_DEVICE(x) amrex::The_Arena()->alloc(x)
#define MY_ALLOC_HOST(x)   amrex::The_Pinned_Arena()->alloc(x)
#define MY_FREE_DEVICE(x)  amrex::The_Arena()->free(x)
#define MY_FREE_HOST(x)    amrex::The_Pinned_Arena()->free(x)
#endif

        auto& dev = amrex::Gpu::Device::syclDevice();
        auto& ctx = amrex::Gpu::Device::syclContext();
        auto& q   = amrex::Gpu::Device::streamQueue(); // AMReX uses orderd queue.

        constexpr int sub_group_size = 16;
        constexpr int group_size = 256;
        constexpr int nitems_per_thread = 4;  // Each thread works on 4 items

        int ngroups = (npts+nitems_per_thread*group_size-1) / (nitems_per_thread*group_size);
        int nthreads_total = ngroups * group_size;

        auto group_result = (T*)MY_ALLOC_DEVICE(sizeof(T)*ngroups);
        auto final_result = (T*)MY_ALLOC_HOST(sizeof(T));

        q.submit([&] (sycl::handler& h)
        {
            sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>
                shared_data(sycl::range<1>(sub_group_size), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads_total),
                                             sycl::range<1>(group_size)),
            [=] (sycl::nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]]
            {
                // thread local reduction
                T x = 0;
                for (int i = item.get_global_linear_id(); i < npts; i += nthreads_total) {
                    x += p[i];
                }
                // For thread 0, r = p[0] + p[nthreads_total  ] + p[2*nthreads_total  ] + ...
                // For thread 1, r = p[1] + p[nthreads_total+1] + p[2*nthreads_total+1] + ...

                // sub-group reduction
                auto const& sg = item.get_sub_group();
                for (int offset = sub_group_size/2; offset > 0; offset /= 2) {
                    T y = sg.shuffle_down(x, offset);
                    x += y;
                }

                // Only the first thread in a sub-group has the full result
                // because of shuffle_down.  It will store the result in
                // shared local memory.
                T* shared = shared_data.get_pointer();
                if (sg.get_local_id()[0] == 0) {
                    shared[sg.get_group_id()[0]] = x;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // The sub-group results are in shared local memory now.
                // The first sub-group in a group will reduce that further
                // for the whole group.
                x = (item.get_local_linear_id() < sg.get_group_range()[0])
                    ? shared[sg.get_local_id()[0]] : T(0);
                if (sg.get_group_id() == 0) {
                    for (int offset = sub_group_size/2; offset > 0; offset /= 2) {
                        T y = sg.shuffle_down(x, offset);
                        x += y;
                    }
                }

                // Now the first thread in a group has the group reduction
                // result.  It will store the result in global memory.
                if (item.get_local_linear_id() == 0) {
                    group_result[item.get_group_linear_id()] = x;
                }
            });
        });

        // We need to launch a second kernel to reduce the group result.
        // This kernel has only one group.

        // In CUDA and HIP, we store the final result in pinned memory.
        // This avoids memcpy or managed momory. But due to a bug, we have
        // to store the result in device memory and then memcpy it back to
        // the host. (Not sure whether or not the bug has been fixed.)
#ifndef AMREX_NO_DPCPP_REDUCE_WORKAROUND
        T* presult = (T*)MY_ALLOC_DEVICE(sizeof(T));
#else
        T* presult = final_result;
#endif

        q.submit([&] (sycl::handler& h)
        {
            sycl::accessor<T, 1, sycl::access::mode::read_write, sycl::access::target::local>
                shared_data(sycl::range<1>(sub_group_size), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(group_size),
                                             sycl::range<1>(group_size)),
            [=] (sycl::nd_item<1> item) [[intel::reqd_sub_group_size(sub_group_size)]]
            {
                // thread local reduction
                T x = 0;
                for (int i = item.get_global_linear_id(); i < ngroups; i += group_size) {
                    x += group_result[i];
                }

                // sub-group reduction
                auto const& sg = item.get_sub_group();
                for (int offset = sub_group_size/2; offset > 0; offset /= 2) {
                    T y = sg.shuffle_down(x, offset);
                    x += y;
                }

                // Only the first thread in a sub-group has the full result
                // because of shuffle_down.  It will store the result in
                // shared local memory.
                T* shared = shared_data.get_pointer();
                if (sg.get_local_id()[0] == 0) {
                    shared[sg.get_group_id()[0]] = x;
                }
                item.barrier(sycl::access::fence_space::local_space);

                // The sub-group results are in shared local memory now.
                // The first sub-group in a group will reduce that further
                // for the whole group.
                x = (item.get_local_linear_id() < sg.get_group_range()[0])
                    ? shared[sg.get_local_id()[0]] : T(0);
                if (sg.get_group_id() == 0) {
                    for (int offset = sub_group_size/2; offset > 0; offset /= 2) {
                        T y = sg.shuffle_down(x, offset);
                        x += y;
                    }
                }

                // Now the first thread in a group has the group reduction
                // result.  It will store the result.
                if (item.get_local_linear_id() == 0) {
                    *presult = x;
                }
            });
        });

#ifndef AMREX_NO_DPCPP_REDUCE_WORKAROUND
        q.submit([&] (sycl::handler& h) { h.memcpy(final_result, presult, sizeof(T)); });
#endif

        q.wait_and_throw();

        T sum = *final_result;

        // free memory
        MY_FREE_DEVICE(group_result);
        MY_FREE_HOST(final_result);
#ifndef AMREX_NO_DPCPP_REDUCE_WORKAROUND
        MY_FREE_DEVICE(presult);
#endif

        double t1 = amrex::second();

        if (n == 0) {
            result.push_back(sum);
            desc.push_back("    amrex2 sum = ");
        }
        ttmp = std::min(ttmp, t1-t0);
    }
    twall.push_back(ttmp);

#endif

    for (int i = 0; i < desc.size(); ++i) {
        amrex::Print() << desc[i] << std::setw(12) << result[i]
                       << ", run time is " << twall[i] << ".\n";
    }
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        int n = 16'777'216;
        {
            ParmParse pp;
            pp.query("n", n);
        }

        amrex::Print() << "Sum of double:\n";
        test_reduce_sum<Real>(n);

        amrex::Print() << "Sum of int:\n";
        test_reduce_sum<int>(n);
    }
    amrex::Finalize();
}
