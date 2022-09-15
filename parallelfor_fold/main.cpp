#include <cassert>
#include <iostream>
#include <string>
#include <vector>

template <int... ctr>
struct CompiletimeOptions {};

template <int ctr>
struct CTOption {
    static constexpr int value = ctr;
};

template <class F>
void ParallelFor(std::size_t N, F&& f)
{
    for (std::size_t i = 0; i < N; ++i) f(i);
}

template <class F, int... Op>
void ParallelFor(std::size_t N, F&& f, CompiletimeOptions<Op...>, int runtime_option)
{
    int option_miss = 0;
    (
        (
            Op == runtime_option ?
              (
                  ParallelFor(N, [f] (std::size_t i) {
                      f(i, CTOption<Op>{});
                  })
              )
            : (
                ++option_miss, void()
              )
        ),...
    );
    assert(option_miss + 1 == sizeof...(Op));
}

int main (int argc, char* argv[])
{
    std::size_t N = 8;
    std::vector<int> va(N,-1);
    std::vector<int> vb(N,-1);
    auto pa = va.data();
    auto pb = vb.data();

    enum all_options: int {
        A0_B0 = 0, A1_B0, A0_B1, A1_B1
    };

    int runtime_option = 0;
    if (argc > 1) {
        runtime_option = std::stoi(std::string(argv[1])) % 4;
    }

    ParallelFor(N, [=] (std::size_t i, auto control)
    {
        if constexpr (control.value == A0_B0 ||
                      control.value == A0_B1) {
            pa[i] = 0;
        }
        if constexpr (control.value == A1_B0 ||
                      control.value == A1_B1) {
            pa[i] = 1;
        }
        if constexpr (control.value == A0_B0 ||
                      control.value == A1_B0) {
            pb[i] = 0;
        }
        if constexpr (control.value == A0_B1 ||
                      control.value == A1_B1) {
            pb[i] = 1;
        }
    },
        CompiletimeOptions<A0_B0, A1_B0, A0_B1, A1_B1>{},
        runtime_option);

    for (std::size_t i = 0; i < N; ++i) {
        std::cout << "  va[" << i << "] = " << va[i]
                  << ", vb[" << i << "] = " << vb[i] << "\n";
    }
}
