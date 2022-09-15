#include <cassert>
#include <iostream>
#include <string>
#include <vector>

template <int... ctr>
struct CompiletimeOptions {};

template <int Value>
using int_c = std::integral_constant<int, Value>;

template <class L> __global__ void launch_global (L f) { f(); }

template <class F>
void ParallelFor(std::size_t N, F&& f)
{
    std::size_t nthreads = 128;
    std::size_t nblocks = (N+nthreads-1)/nthreads;
    launch_global<<<nblocks, nthreads, 0, 0>>>([=] __device__ ()
    {
        std::size_t i = blockDim.x*blockIdx.x+threadIdx.x;
        if (i < N) f(i);
    });
}

template <class F, int... As, int... Bs>
void ParallelFor(std::size_t N, F&& f, CompiletimeOptions<As...>, int A_option,
                 CompiletimeOptions<Bs...>, int B_option)
{
    int option_miss = 0;
    (
        ([=,&option_miss] (auto B) {
            static constexpr int b = B();
            ((
            ((As == A_option) && (b == B_option)) ?
                ParallelFor(N, [f] __device__ (std::size_t i) {
                    f(i, int_c<As>{}, int_c<b>{});
                })
            : (
                ++option_miss, void()
              )
            ), ...);
        }(int_c<Bs>{}), ...)
    );
    assert(option_miss + 1 == sizeof...(As) * sizeof...(Bs));
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

#if 0
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
#else
    ParallelFor(N, [=] __device__ (std::size_t i)
    {
        pa[i] = 1;
    });
#endif

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
