#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_VisMF.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
	MultiFab mf;
	VisMF::Read(mf, "inmf");
	VisMF::Write(mf, "outmf");
    }
    amrex::Finalize();
}
