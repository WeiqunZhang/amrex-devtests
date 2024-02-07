#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Random.H>
#include <AMReX_Reduce.H>
#include <AMReX_Partition.H>

using namespace amrex;

template <typename T, typename P>
int new_partition (Gpu::DeviceVector<T>& dv, P&& is_left)
{
    const int n = dv.size();
    if (n == 0) { return 0; }

    auto* pv = dv.data();

    const int num_left = Reduce::Sum<int>(n,
        [=] AMREX_GPU_DEVICE (int i) -> int
        {
            return int(is_left(pv[i]));
        });

    const int num_swaps = std::min(num_left, n - num_left);
    if (num_swaps == 0) { return num_left; }

    Gpu::DeviceVector<int> index_left(num_swaps);
    Gpu::DeviceVector<int> index_right(num_swaps);
    int * const p_index_left = index_left.dataPtr();
    int * const p_index_right = index_right.dataPtr();

    Scan::PrefixSum<int>(n,
        [=] AMREX_GPU_DEVICE (int i) -> int
        {
            return int(!is_left(pv[i]));
        },
        [=] AMREX_GPU_DEVICE (int i, int const& s)
        {
            if (!is_left(pv[i])) {
                int dst = s;
                if (dst < num_swaps) {
                    p_index_right[dst] = i;
                }
            } else {
                int dst = num_left-1-(i-s);
                if (dst < num_swaps) {
                    p_index_left[dst] = i;
                }
            }
        },
        Scan::Type::exclusive, Scan::noRetSum);

    // There are at most num_swaps.
    //
    // p_index_right stores the pre-swap index of the first num_swaps
    // particles that belong to the right partition.
    //
    // p_index_left store the pre-swap index of the last num_swaps particls
    // that belong to the left partition.
    //
    // Not all marked particles need to be swapped.

    ParallelFor(num_swaps,
        [=] AMREX_GPU_DEVICE (int i)
        {
            int left_i = p_index_left[i];
            int right_i = p_index_right[i];
            if (right_i < left_i) {
                amrex::Swap(pv[left_i], pv[right_i]);
            }
        });

    Gpu::streamSynchronize(); // for index_left and index_right

    return num_left;
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        unsigned int max_size = 100'000'000u;

        Gpu::DeviceVector<unsigned int> iv0(max_size);
        Gpu::DeviceVector<unsigned int> iv1(max_size);

        auto* p0 = iv0.data();
        ParallelForRNG(max_size, [=] AMREX_GPU_DEVICE (unsigned i, RandomEngine const& engine)
        {
            p0[i] = Random_int(std::numeric_limits<unsigned int>::max(), engine);
        });
        Gpu::streamSynchronize();

        auto flag = [=] AMREX_GPU_DEVICE (unsigned int i) { return (i%10) != 0; };

        double told;
        for (int n = 0; n < 2; ++n) {
            Gpu::copyAsync(Gpu::deviceToDevice, iv0.cbegin(), iv0.cend(),
                           iv1.begin());
            Gpu::streamSynchronize();
            double t0 = amrex::second();
            auto nleft = Partition(iv1, flag);
            told = amrex::second() - t0;
            amrex::Print() << "Number of particles on the left: " << nleft << "\n";
        }

        double tnew;
        for (int n = 0; n < 2; ++n) {
            Gpu::copyAsync(Gpu::deviceToDevice, iv0.cbegin(), iv0.cend(),
                           iv1.begin());
            Gpu::streamSynchronize();
            double t0 = amrex::second();
            auto nleft = new_partition(iv1, flag);
            tnew = amrex::second() - t0;
            amrex::Print() << "Number of particles on the left: " << nleft << "\n";
        }

        amrex::Print() << "Old amrex::Partition: " << told << "\n"
                       << "New amrex::Partition: " << tnew << "\n";
    }
    amrex::Finalize();
}
