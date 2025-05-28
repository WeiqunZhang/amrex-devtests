#include <AMReX.H>
#include <AMReX_Parser.H>
#include <map>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
	Parser parser("avg_curr = abs(-8e-9 * clight / bunch_length1);"
		      "head_curr = 0.1e3;"
		      "diff_curr = avg_curr - head_curr;"
		      "a = (diff_curr / (bunch_length1/2)) / avg_curr;"
		      "b = 50e-6;"
		      "ylo = -bunch_length1/2;"
		      "yhi = bunch_length1/2;"
		      "lower = b*(sqrt(pi/2)*(1 - a*z)*erf(((1/sqrt(2))*(ylo - z))/b) + a*b*exp(-(0.5*(ylo - z)^2)/b^2));"
		      "upper = b*(sqrt(pi/2)*(1 - a*z)*erf(((1/sqrt(2))*(yhi - z))/b) + a*b*exp(-(0.5*(yhi - z)^2)/b^2));"
		      "upper - lower");

	parser.registerVariables({"z"});
	parser.setConstant("clight", 3.e8);
	parser.setConstant("bunch_length1", 1.24e-2);
        parser.setConstant("pi", 3.14);

        parser.print();
        amrex::Print() << "depth = " << parser.depth() << std::endl;

        auto const exe = parser.compile<1>();
        amrex::Print() << "max stack size = " << parser.maxStackSize() << std::endl;
	parser.printExe();

    }
    amrex::Print() << "\n";

    amrex::Finalize();
}
