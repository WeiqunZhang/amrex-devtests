#include <AMReX.H>
#include <AMReX_FArrayBox.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv, false);
    {
        int N = 8;
        Gpu::ManagedVector<int> va(N, -1);
        Gpu::ManagedVector<int> vb(N, -1);
        int* pa = va.data();
        int* pb = vb.data();

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

        ParallelFor(N, [=] AMREX_GPU_DEVICE (int i, auto A_control, auto B_control)
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
            CompileTimeOptions<A0,A1>{}, A_runtime_option,
            CompileTimeOptions<B0,B1>{}, B_runtime_option);
        
        Gpu::synchronize();

        for (int i = 0; i < N; ++i) {
            std::cout << "  pa[" << i << "] = " << pa[i]
                      << ", pb[" << i << "] = " << pb[i] << "\n";
        }

        ParallelFor(N, [=] AMREX_GPU_DEVICE (int i, auto control)
        {
            auto lpa = pa;
            if constexpr (control.value == A0) {
                lpa[i] = 30;       
            }
            if constexpr (control.value == A1) {
                lpa[i] = 31;       
            }
        },
            CompileTimeOptions<A0,A1>{}, A_runtime_option);

        Gpu::synchronize();

        for (int i = 0; i < N; ++i) {
            std::cout << "  pa[" << i << "] = " << pa[i] << "\n";
        }

        Box box(IntVect(0),IntVect(31));
        FArrayBox fab1(box);
        FArrayBox fab2(box,2);
        auto const& a1 = fab1.array();
        auto const& a2 = fab2.array();

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k, auto A_control, auto B_control)
        {
            auto const& la1 = a1;
            if constexpr (A_control.value == 1 && B_control.value == 1) {
                la1(i,j,k) = 11;
            } else {
                la1(i,j,k) = 10;
            }
        },
            CompileTimeOptions<A0,A1>{}, A_runtime_option,
            CompileTimeOptions<B0,B1>{}, B_runtime_option);

        ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n,
                                                  auto A_control, auto B_control)
        {
            auto const& la2 = a2;
            if constexpr (A_control.value == 1 && B_control.value == 1) {
                la2(i,j,k,n) = 11;
            } else {
                la2(i,j,k,n) = 10;
            }
        },
            CompileTimeOptions<A0,A1>{}, A_runtime_option,
            CompileTimeOptions<B0,B1>{}, B_runtime_option);

        amrex::Print() << " fab1.sum = " << fab1.sum(0) << " fab2.sum = " << fab2.sum(1) << std::endl;

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k, auto A_control)
        {
            auto const& la1 = a1;
            if constexpr (A_control.value == 1) {
                la1(i,j,k) = 11;
            } else {
                la1(i,j,k) = 10;
            }
        },
            CompileTimeOptions<A0,A1>{}, A_runtime_option);

        ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n, auto A_control)
        {
            auto const& la2 = a2;
            if constexpr (A_control.value == 1) {
                la2(i,j,k,n) = 11;
            } else {
                la2(i,j,k,n) = 10;
            }
        },
            CompileTimeOptions<A0,A1>{}, A_runtime_option);

        amrex::Print() << " fab1.sum = " << fab1.sum(0) << " fab2.sum = " << fab2.sum(1) << std::endl;
    }
    amrex::Finalize();
}
