#include <CL/sycl.hpp>

#include <chrono>
#include <iostream>
#include <limits>
#include <random>

#define AMREX_FORCE_INLINE inline __attribute__((always_inline))
#define AMREX_RESTRICT __restrict__

template <typename T>
struct STVA
{
    char status;
    T value;
};

template <typename T>
struct BlockStatus
{
    template<typename U>
    union Data {
        STVA<U> s;
        uint64_t i;
        void operator=(Data<U> const&) = delete;
        void operator=(Data<U> &&) = delete;
    };
    Data<T> d;

    AMREX_FORCE_INLINE
    void write (char a_status, T a_value) {
        Data<T> tmp;
        tmp.s = {a_status, a_value};
        typedef unsigned long long ULL;
        static_assert(sizeof(ULL) == sizeof(Data<T>));
        constexpr auto as = sycl::access::address_space::global_space;
        sycl::atomic<ULL, as> a{sycl::multi_ptr<ULL,as>(reinterpret_cast<ULL*>(&d))};
        a.exchange(reinterpret_cast<unsigned long long&>(tmp),
                   sycl::memory_order::relaxed);
    }

    AMREX_FORCE_INLINE
    T get_aggregate() const { return d.s.value; }

    AMREX_FORCE_INLINE
    STVA<T> read () volatile {
        typedef unsigned long long ULL;
        static_assert(sizeof(ULL) == sizeof(Data<T>));
        constexpr auto as = sycl::access::address_space::global_space;
        sycl::atomic<ULL, as> a{sycl::multi_ptr<ULL,as>
                                (reinterpret_cast<ULL*>(const_cast<Data<T>*>(&d)))};
        ULL tmp = a.load(sycl::memory_order::relaxed);
        return (*reinterpret_cast<Data<T>*>(&tmp)).s;
    }

    AMREX_FORCE_INLINE
    void set_status (char a_status) { d.s.status = a_status; }

    AMREX_FORCE_INLINE
    STVA<T> wait () volatile {
        STVA<T> r;
        do {
            sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            r = read();
        } while (r.status == 'x');
        return r;
    }
};

inline std::size_t align (std::size_t size)
{
    constexpr std::size_t align_requirement = 16;
    return ((size + (align_requirement-1)) / align_requirement) * align_requirement;
}


template <typename T, typename INT, typename FIN, typename FOUT>
T exclusive_scan (INT N, FIN && fin, FOUT && fout, sycl::device const& device,
                  sycl::context const& context, sycl::queue& q)
{
    constexpr int warp_size = 16;
    constexpr int nwarps_per_block = 8;
    constexpr int nthreads = nwarps_per_block*warp_size;
    constexpr int nchunks = 12;
    constexpr int nelms_per_block = nthreads * nchunks;
    int nblocks = (static_cast<long>(N) + nelms_per_block - 1) / nelms_per_block;
    std::size_t sm = sizeof(T) * (warp_size + nwarps_per_block) + sizeof(int);

    static_assert(sizeof(STVA<T>) <= 8);

    std::size_t nbytes_blockstatus = align(sizeof(BlockStatus<T>)*nblocks);
    std::size_t nbytes_blockid = align(sizeof(unsigned int));
    std::size_t nbytes_totalsum = align(sizeof(T));
    auto dp = (char*)sycl::malloc_shared(nbytes_blockstatus + nbytes_blockid + nbytes_totalsum,
                                         device, context);
    BlockStatus<T>* AMREX_RESTRICT block_status_p = (BlockStatus<T>*)dp;
    unsigned int* AMREX_RESTRICT virtual_block_id_p = (unsigned int*)(dp + nbytes_blockstatus);
    T* AMREX_RESTRICT totalsum_p = (T*)(dp + nbytes_blockstatus + nbytes_blockid);

    q.submit([&](sycl::handler& h)
    {
        h.parallel_for(sycl::range<1>(nblocks), [=] (sycl::item<1> item)
        {
            int i = item.get_linear_id();
            BlockStatus<T>& block_status = block_status_p[i];
            block_status.set_status('x');
            if (i == 0) {
                *virtual_block_id_p = 0;
                *totalsum_p = 0;
            }
        });
    });

    q.submit([&] (sycl::handler& h)
    {
        sycl::accessor<unsigned long long, 1, sycl::access::mode::read_write,
                       sycl::access::target::local>
            shared_data(sycl::range<1>(sm), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(nthreads*nblocks),
                                         sycl::range<1>(nthreads)),
        [=] (sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]]
        {
            T* shared = (T*)shared_data.get_pointer().get();
            T* shared2 = shared + warp_size;

            auto const& sg = item.get_sub_group();
            int lane = sg.get_local_id()[0];
            int warp = sg.get_group_id()[0];
            int nwarps = sg.get_group_range()[0];

            int threadIdxx = item.get_local_id(0);
            int blockDimx = item.get_local_range(0);
            int gridDimx = item.get_group_range(0);

            // First of all, get block virtual id.  We must do this to
            // avoid deadlock because blocks may be launched in any order.
            // Anywhere in this function, we should not use blockIdx.
            int virtual_block_id = 0;
            if (gridDimx > 1) {
                int& virtual_block_id_shared = *((int*)(shared2+nwarps));
                if (threadIdxx == 0) {
                    constexpr auto as = sycl::access::address_space::global_space;
                    sycl::atomic<unsigned int, as> a{sycl::multi_ptr<unsigned int, as>(virtual_block_id_p)};
                    virtual_block_id_shared = a.fetch_add(1u);
                }
                item.barrier(sycl::access::fence_space::local_space);
                virtual_block_id = virtual_block_id_shared;
            }

            // Each block processes [ibegin,iend).
            INT ibegin = nelms_per_block * virtual_block_id;
            INT iend = std::min(static_cast<INT>(ibegin+nelms_per_block), N);
            BlockStatus<T>& block_status = block_status_p[virtual_block_id];

            //
            // The overall algorithm is based on "Single-pass Parallel
            // Prefix Scan with Decoupled Look-back" by D. Merrill &
            // M. Garland.
            //

            // Each block is responsible for nchunks chunks of data,
            // where each chunk has blockDim.x elements, one for each
            // thread in the block.
            T sum_prev_chunk = 0; // inclusive sum from previous chunks.
            T tmp_out[nchunks]; // block-wide inclusive sum for chunks
            for (int ichunk = 0; ichunk < nchunks; ++ichunk) {
                INT offset = ibegin + ichunk*blockDimx;
                if (offset >= iend) break;

                offset += threadIdxx;
                T x0 = (offset < iend) ? fin(offset) : 0;
                if (offset == N-1) {
                    *totalsum_p += x0;
                }
                T x = x0;
                // Scan within a warp
                for (int i = 1; i <= warp_size; i *= 2) {
                    T s = sg.shuffle_up(x, i);
                    if (lane >= i) x += s;
                }

                // x now holds the inclusive sum within the warp.  The
                // last thread in each warp holds the inclusive sum of
                // this warp.  We will store it in shared memory.
                if (lane == warp_size - 1) {
                    shared[warp] = x;
                }

                item.barrier(sycl::access::fence_space::local_space);

                // The first warp will do scan on the warp sums for the
                // whole block.
                if (warp == 0) {
                    T y = (lane < nwarps) ? shared[lane] : 0;
                    for (int i = 1; i <= warp_size; i *= 2) {
                        T s = sg.shuffle_up(y, i);
                        if (lane >= i) y += s;
                    }

                    if (lane < nwarps) shared2[lane] = y;
                }

                item.barrier(sycl::access::fence_space::local_space);

                // shared[0:nwarps) holds the inclusive sum of warp sums.

                // Also note x still holds the inclusive sum within the
                // warp.  Given these two, we can compute the inclusive
                // sum within this chunk.
                T sum_prev_warp = (warp == 0) ? 0 : shared2[warp-1];
                tmp_out[ichunk] = sum_prev_warp + sum_prev_chunk + (x-x0);
                sum_prev_chunk += shared2[nwarps-1];
            }

            // sum_prev_chunk now holds the sum of the whole block.
            if (threadIdxx == 0 && gridDimx > 1) {
                block_status.write((virtual_block_id == 0) ? 'p' : 'a',
                                   sum_prev_chunk);
            }

            if (virtual_block_id == 0) {
                for (int ichunk = 0; ichunk < nchunks; ++ichunk) {
                    INT offset = ibegin + ichunk*blockDimx + threadIdxx;
                    if (offset >= iend) break;
                    fout(offset, tmp_out[ichunk]);
                    if (offset == N-1) {
                        *totalsum_p += tmp_out[ichunk];
                    }
                }
            } else if (virtual_block_id > 0) {

                if (warp == 0) {
                    T exclusive_prefix = 0;
                    BlockStatus<T> volatile* pbs = block_status_p;
                    for (int iblock0 = virtual_block_id-1; iblock0 >= 0; iblock0 -= warp_size)
                    {
                        int iblock = iblock0-lane;
                        STVA<T> stva{'p', 0};
                        if (iblock >= 0) {
                            stva = pbs[iblock].wait();
                        }

                        T x = stva.value;

                        // implement our own __ballot
                        unsigned status_bf = (stva.status == 'p') ? (0x1u << lane) : 0;
                        for (int i = 1; i < warp_size; i *= 2) {
                            status_bf |= sg.shuffle_xor(status_bf, i);
                        }

                        bool stop_lookback = status_bf & 0x1u;
                        if (stop_lookback == false) {
                            if (status_bf != 0) {
                                T y = x;
                                if (lane > 0) x = 0;
                                unsigned int bit_mask = 0x1u;
                                for (int i = 1; i < warp_size; ++i) {
                                    bit_mask <<= 1;
                                    if (i == lane) x = y;
                                    if (status_bf & bit_mask) {
                                        stop_lookback = true;
                                        break;
                                    }
                                }
                            }

                            for (int i = warp_size/2; i > 0; i /= 2) {
                                x += sg.shuffle_down(x,i);
                            }
                        }

                        if (lane == 0) { exclusive_prefix += x; }
                        if (stop_lookback) break;
                    }

                    if (lane == 0) {
                        block_status.write('p', block_status.get_aggregate() + exclusive_prefix);
                        shared[0] = exclusive_prefix;
                    }
                }

                item.barrier(sycl::access::fence_space::local_space);

                T exclusive_prefix = shared[0];

                for (int ichunk = 0; ichunk < nchunks; ++ichunk) {
                    INT offset = ibegin + ichunk*blockDimx + threadIdxx;
                    if (offset >= iend) break;
                    T t = tmp_out[ichunk] + exclusive_prefix;
                    fout(offset, t);
                    if (offset == N-1) {
                        *totalsum_p += t;
                    }
                }
            }
        });
    });

    T totalsum;
    T* hp = &totalsum;
    q.submit([&] (sycl::handler& h) { h.memcpy(hp, totalsum_p, sizeof(T)); });

    q.wait();

    sycl::free(dp, context);

    return totalsum;
}

int main (int argc, char* argv[])
{
    sycl::gpu_selector device_selector;
    sycl::platform platform(device_selector);
    auto const& gpu_devices = platform.get_devices();
    if (gpu_devices.empty()) {
        std::cout << "No GPU device found\n";
        return 1;
    }
    auto const& device = gpu_devices[0];
    sycl::context context(device);
    sycl::queue q(context, device, sycl::property_list{sycl::property::queue::in_order{}});

    std::mt19937 gen;
    std::uniform_int_distribution<> dis(10000, 200000000);

    using MaxResSteadyClock = std::conditional<std::chrono::high_resolution_clock::is_steady,
                                               std::chrono::high_resolution_clock,
                                               std::chrono::steady_clock>::type;

    // xxxxxx
    int tmax = 5; // minutes;
    int debug = true;

    auto t0 = MaxResSteadyClock::now();
    auto t1 = t0;
    auto t2 = t0;

    const long ntests = std::numeric_limits<long>::max();
    for (long itest = 0; itest < ntests; ++itest) {
        int N = dis(gen);
        int* p = (int*)sycl::malloc_shared(sizeof(int)*N, device, context);
        int* p2 = (int*)sycl::malloc_shared(sizeof(int)*N, device, context);

        q.submit([&](sycl::handler& h)
        {
            h.parallel_for(sycl::range<1>(N), [=] (sycl::item<1> item)
            {
                p[item.get_linear_id()] = 1;
            });
        });

        typedef int T;
        typedef int INT;
        T sum = exclusive_scan<T>(N,
                                  [=] (INT i) { return p[i]; },
                                  [=] (INT i, T result) { p2[i] = result; },
                                  device, context, q);

        int* hp = (int*)sycl::malloc_host(sizeof(int)*N, context);
        q.submit([&] (sycl::handler& h) { h.memcpy(hp, p2, sizeof(int)*N); });
        q.wait();

        for (int i = 0; i < N; ++i) {
            if (hp[i] != i) {
                std::cout << "N = " << N << " failed" << std::endl;
                exit(1);
            }
        }

        sycl::free(p, context);
        sycl::free(p2, context);
        sycl::free(hp, context);

        if (itest == 0) {
            std::cout << "First test passed" << std::endl;
        }

        t2 = MaxResSteadyClock::now();
        auto dt = std::chrono::duration_cast<std::chrono::minutes>(t2-t1).count();
        auto dt2 = std::chrono::duration_cast<std::chrono::minutes>(t2-t0).count();
        if (debug) {
            std::cout << "Test #" << itest << ": N = " << N << " passed." << std::endl;
        }
        if (dt >= 1) {
            std::swap(t1,t2);
            std::cout << "After " << dt2 << " minutes, " << itest+1 << " tests have passed."
                      << std::endl;
        }
        if (dt2 > tmax) {
            std::cout << "Finished " << itest+1 << " tests in " << dt2 << " minutes."
                      << std::endl;
            break;
        }
    }
}
