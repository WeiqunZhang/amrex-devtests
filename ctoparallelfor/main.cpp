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
        Gpu::ManagedVector<int> vc(N, -1);
        int* pa = va.data();
        int* pb = vb.data();
        int* pc = vc.data();

        enum A_options: int {
            A0 = 0, A1
        };

        enum B_options: int {
            B0 = 0, B1, B2, B3
        };

        enum C_options: int {
            C0 = 0, C1
        };

        for (int Ai = 0; Ai < 2; ++Ai) {
        for (int Bi = 0; Bi < 4; ++Bi) {
        for (int Ci = 0; Ci < 2; ++Ci) {

            ParallelFor(N, [=] AMREX_GPU_DEVICE (int i, auto A_control, auto B_control,
                                                 auto C_control)
            {
                auto lpa = pa; // nvcc limitation
                auto lpb = pb;
                auto lpc = pc;
                if constexpr (A_control.value == A0) {
                    lpa[i] = 0;
                } else if constexpr (A_control.value == A1) {
                    lpa[i] = 1;
                }
                if constexpr (B_control.value == B0) {
                    lpb[i] = 0;
                } else if constexpr (B_control.value == B1) {
                    lpb[i] = 1;
                } else if constexpr (B_control.value == B2) {
                    lpb[i] = 2;
                } else if constexpr (B_control.value == B3) {
                    lpb[i] = 3;
                }
                if constexpr (C_control.value == C0) {
                    lpc[i] = 0;
                } else if constexpr (C_control.value == C1) {
                    lpc[i] = 1;
                }
            },
            TypeList<CompileTimeOptions<A0,A1>,
                     CompileTimeOptions<B0,B1,B2,B3>,
                     CompileTimeOptions<C0,C1>>{},
            {Ai, Bi, Ci});
        
            Gpu::synchronize();
            std::cout << "  a = " << pa[0]
                      << ", b = " << pb[0]
                      << ", c = " << pc[0] << "\n";
        }}}


        for (int Ai = 0; Ai < 2; ++Ai) {
        for (int Bi = 0; Bi < 4; ++Bi) {
            ParallelFor(N, [=] AMREX_GPU_DEVICE (int i, auto A_control, auto B_control)
            {
                auto lpa = pa; // nvcc limitation
                auto lpb = pb;
                if constexpr (A_control.value == A0) {
                    lpa[i] = 10;
                } else if constexpr (A_control.value == A1) {
                    lpa[i] = 11;
                }
                if constexpr (B_control.value == B0) {
                    lpb[i] = 10;
                } else if constexpr (B_control.value == B1) {
                    lpb[i] = 11;
                } else if constexpr (B_control.value == B2) {
                    lpb[i] = 12;
                } else if constexpr (B_control.value == B3) {
                    lpb[i] = 13;
                }
            },
                TypeList<CompileTimeOptions<A0,A1>,
                         CompileTimeOptions<B0,B1,B2,B3>>{},
                {Ai,Bi});
        
            Gpu::synchronize();
            std::cout << "  a = " << pa[0]
                      << ", b = " << pb[0] << "\n";

        }}

        for (int Ai = 0; Ai < 2; ++Ai) {
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
                TypeList<CompileTimeOptions<A0,A1>>{},
                {Ai});

            Gpu::synchronize();
            std::cout << "  a = " << pa[0] << "\n";
        }

        Box box(IntVect(0),IntVect(31));
        FArrayBox fab1(box);
        FArrayBox fab2(box,2);
        auto const& a1 = fab1.array();
        auto const& a2 = fab2.array();

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k,
                                               auto A_control, auto B_control)
        {
            auto const& la1 = a1;
            if constexpr (A_control.value == 1 && B_control.value == 1) {
                la1(i,j,k) = 11;
            } else {
                la1(i,j,k) = 10;
            }
        },
            TypeList<CompileTimeOptions<A0,A1>,
                     CompileTimeOptions<B0,B1>>{},
            {A1, B1});

        amrex::Print() << "  fab1.sum(0) = " << fab1.sum(0)
                       << " expected value = " << box.numPts()*11 << "\n";

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
            TypeList<CompileTimeOptions<A0,A1>,
                     CompileTimeOptions<B0,B1>>{},
            {A1, B0});

        amrex::Print() << " fab2.sum(1) = " << fab2.sum(1)
                       << " expected value = " << box.numPts()*10 << "\n";

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k, auto A_control)
        {
            auto const& la1 = a1;
            if constexpr (A_control.value == 1) {
                la1(i,j,k) = 101;
            } else {
                la1(i,j,k) = 100;
            }
        },
            TypeList<CompileTimeOptions<A0,A1>>{},
            {A1});

        amrex::Print() << "  fab1.sum(0) = " << fab1.sum(0)
                       << " expected value = " << box.numPts()*101 << "\n";

        ParallelFor(box, 2, [=] AMREX_GPU_DEVICE (int i, int j, int k, int n,
                                                  auto A_control)
        {
            auto const& la2 = a2;
            if constexpr (A_control.value == 1) {
                la2(i,j,k,n) = 201;
            } else {
                la2(i,j,k,n) = 200;
            }
        },
            TypeList<CompileTimeOptions<A0,A1>>{},
            {A0});

        amrex::Print() << " fab2.sum(1) = " << fab2.sum(1)
                       << " expected value = " << box.numPts()*200 << "\n";
    }
    amrex::Finalize();
}
