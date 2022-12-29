#include <AMReX.H>
#include <AMReX_Gpu.H>

using namespace amrex;

template <typename F>
void test (F && f)
{
    if constexpr (amrex::IsCallable<F,int>::value) {
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) { f(1); } );
    } else {
        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int) { f(1,2); } );
    }
}

AMREX_GPU_DEVICE void print (int i)
{
#if AMREX_DEVICE_COMPILE
    AMREX_DEVICE_PRINTF("i = %d\n", i);
#else
    std::printf("i = %d\n", i);
#endif
}

AMREX_GPU_DEVICE void print (int i, int j)
{
#if AMREX_DEVICE_COMPILE
    AMREX_DEVICE_PRINTF("i = %d, j = %d\n", i, j);
#else
    std::printf("i = %d, j = %d\n", i, j);
#endif
}

struct Fn
{   
    AMREX_GPU_DEVICE void operator() (int i) const { print(i); }
};

struct Fn2
{   
    AMREX_GPU_DEVICE void operator() (int i, int j) const { print(i,j); }
};

int main(int argc, char* argv[])
{   
    amrex::Initialize(argc,argv);
    {
        int k = 1000;

        test([ ] AMREX_GPU_DEVICE (int i) { print(i); });
        test([=] AMREX_GPU_DEVICE (int i) { print(i+k); });
        test(Fn());

        test([ ] AMREX_GPU_DEVICE (int i, int j) { print(i,j); });
        test([=] AMREX_GPU_DEVICE (int i, int j) { print(i+k,j); });
        test(Fn2());
    }
    amrex::Finalize();
}
