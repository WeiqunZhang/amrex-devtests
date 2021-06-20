#include <AMReX.H>
#include <AMReX_Parser.H>
#include <map>

using namespace amrex;

void test (std::string const& f, std::map<std::string,Real> const& constants,
           Vector<std::string> const& variables)
{
    amrex::Print() << "\n" << f << "\n";
    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.registerVariables(variables);
    parser.print();
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        test("a*sin(pi*x) + b*cos(2*pi*y+pi*z)",
             {{"a", 2.5}, {"b", -3.6}, {"pi", 3.1415926535897932}},
             {"x","y","z"});
        test("if(z<zp, nc*exp((z-zc)/lgrad), if(z<=zp2, 2.*nc, nc*exp(-(z-zc2)/lgrad)))",
             {{"zp", 1.0}, {"nc", 1.e-6}, {"zc", 0.6}, {"lgrad", 2.5}, {"zp2", 2.0}, {"zc2", 1.3}},
             {"z"});
        test("3.*2.*x*.4*0.5*sin(y)*2.*3.*z*4.*0.5",
             {},
             {"x","y","z"});
    }
    amrex::Print() << "\n";



    amrex::Finalize();
}
