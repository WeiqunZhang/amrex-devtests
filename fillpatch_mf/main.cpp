#include <main.H>

int main(int argc, char* argv[])
{

  // Initialize AMReX
  amrex::Initialize(argc,argv);

  {

  amrex::Print() << "\n--------------------------\n";
  amrex::Print() << "FillPatch Mini-App\n";
  amrex::Print() << "--------------------------\n";

  AmrCoreDerived acd;
  acd.InitData();
  acd.FillPatch();
  amrex::Real tstart = amrex::second();
  acd.FillPatch();
  amrex::Print() << "-------------------------------------\n";
  amrex::Print() << "FillPatch Time (sec) = " << amrex::second() - tstart << std::endl;
  amrex::Print() << "-------------------------------------\n\n";

  }

  // Finalize AMReX
  amrex::Finalize();

  return 0;
}
