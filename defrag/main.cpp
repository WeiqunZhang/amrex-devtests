#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
	Vector<Gpu::DeviceVector<double>> dv(16);
	for (auto& v : dv) { v.resize(1000000); }
	for (int i = 0; i < 100; ++i) {
	    double factor = 1.01;
	    for (auto& v : dv) {
		v.resize(v.size()*factor);
	    }
	}
	Arena::PrintUsage();
    }
    amrex::Finalize();
}
