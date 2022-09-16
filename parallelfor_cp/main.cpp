#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#include <hip/hip_runtime.h>
#endif

template <int... ctr>
struct CompiletimeOptions {};

template <int V>
using IntConst = std::integral_constant<int, V>;

namespace meta {

    template <typename... Ts>
    struct List {};

    template <typename... Ts>
    static constexpr List<Ts...> list{};

    template <typename... As, typename... Bs>
    constexpr List<As..., Bs...> operator+ (List<As...>, List<Bs...>) { return {}; }

    template <typename A, typename... Bs>
    constexpr List<List<A,Bs>...> product (A, List<Bs...> b) { return {}; }

    template <typename... As, typename... Bs>
    constexpr auto cartesian_product (List<As...>, List<Bs...> b) {
        return (list<> + ... + product(As{}, b));
    }
}

template <class L> __global__ void launch_global (L f) { f(); }

template <class F>
void ParallelFor (std::size_t N, F&& f)
{
    std::size_t nthreads = 128;
    std::size_t nblocks = (N+nthreads-1)/nthreads;
    launch_global<<<nblocks, nthreads, 0, 0>>>([=] __device__ ()
    {
        std::size_t i = blockDim.x*blockIdx.x+threadIdx.x;
        if (i < N) f(i);
    });
}

template <class F, int A, int B>
bool ParallelFor_helper2 (std::size_t N, F&& f, meta::List<IntConst<A>, IntConst<B>>, int Ao, int Bo)
{
    if (A == Ao && B == Bo) {
        ParallelFor(N, [f] __device__ (std::size_t i)
        {
            f(i, IntConst<A>{}, IntConst<B>{});
        });
        return true;
    } else {
        return false;
    }
}

template <class F, typename... CTOs>
void ParallelFor_helper1 (std::size_t N, F&& f, meta::List<CTOs...>, int Ao, int Bo)
{
    bool found_option = (false || ... || ParallelFor_helper2(N, std::forward<F>(f), CTOs{}, Ao, Bo));
    assert(found_option);
}

template <class F, int... As, int... Bs>
void ParallelFor (std::size_t N, F&& f,
                  CompiletimeOptions<As...>, int A_option,
                  CompiletimeOptions<Bs...>, int B_option)
{
    using AL = meta::List<IntConst<As>...>;
    using BL = meta::List<IntConst<Bs>...>;
    using CTOs = decltype(meta::cartesian_product(AL{}, BL{}));
    ParallelFor_helper1(N, std::forward<F>(f), CTOs{}, A_option, B_option);
}

int main (int argc, char* argv[])
{
    std::size_t N = 8;
    int *pa, *pb;
#ifdef __CUDACC__
    cudaMallocManaged(&pa, N*sizeof(int));
    cudaMallocManaged(&pb, N*sizeof(int));
#else
    hipMallocManaged(&pa, N*sizeof(int));
    hipMallocManaged(&pb, N*sizeof(int));
#endif

    ParallelFor(N, [=] __device__ (std::size_t i)
    {
        pa[i] = -1;
        pb[i] = -1;
    });

    enum A_options: int {
        A0 = 0, A1
    };

    enum B_options: int {
        B0 = 0, B1
    };

    int A_runtime_option = 0;
    int B_runtime_option = 0;
    if (argc > 2) {
        A_runtime_option = std::stoi(std::string(argv[1])) % 2;
        B_runtime_option = std::stoi(std::string(argv[2])) % 2;
    }

    ParallelFor(N, [=] __device__ (std::size_t i, auto A_control, auto B_control)
    {
        auto lpa = pa; // nvcc limitation
        auto lpb = pb;
        if constexpr (A_control.value == A0) {
            lpa[i] = 0;
        } else if constexpr (A_control.value == A1) {
            lpa[i] = 1;
        }
        if constexpr (B_control.value == B0) {
            lpb[i] = 0;
        } else if constexpr (B_control.value == B1) {
            lpb[i] = 1;
        }
    },
        CompiletimeOptions<A0,A1>{},
        A_runtime_option,
        CompiletimeOptions<B0,B1>{},
        B_runtime_option);

#ifdef __CUDACC__
    cudaDeviceSynchronize();
#else
    hipDeviceSynchronize();
#endif

    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "  pa[" << i << "] = " << pa[i]
                  << ", pb[" << i << "] = " << pb[i] << "\n";
    }

#ifdef __CUDACC__
    cudaFree(pa);
    cudaFree(pb);
#else
    hipFree(pa);
    hipFree(pb);
#endif
}
