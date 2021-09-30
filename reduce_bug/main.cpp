#include <AMReX.H>
#include <AMReX_Reduce.H>

using namespace amrex;

void test ()
{
    Gpu::DeviceVector<int> v(256*256*256, 1);        
    int sum = Reduce::Sum(v.size(), v.data());
    amrex::Print() << " sum = " << sum << std::endl;
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(v.size() == sum, "Reduce::Sum() failed");
}

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        for (int itest = 0; itest < 10; ++itest) {
            test();
        }
    }
    amrex::Finalize();
}
