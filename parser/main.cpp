#include <AMReX.H>
#include <AMReX_Parser.H>
#include <map>

using namespace amrex;

void test0 (std::string const& f,
            std::map<std::string,Real> const& constants)
{
    amrex::Print() << "\n" << f << "\n";
    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.print();
    amrex::Print() << "depth = " << parser.depth() << std::endl;
    auto const exe = parser.compile<0>();
    amrex::Print() << "f() = " << exe() << std::endl;
}

template <typename F>
void test3 (std::string const& f,
            std::map<std::string,Real> const& constants,
            Vector<std::string> const& variables,
            F && fb,
            Array<Real,3> const& lo, Array<Real,3> const& hi)
{
    amrex::Print() << "\n" << f << "\n";
    Parser parser(f);
    for (auto const& kv : constants) {
        parser.setConstant(kv.first, kv.second);
    }
    parser.registerVariables(variables);
    parser.print();
    amrex::Print() << "depth = " << parser.depth() << std::endl;

    auto const exe = parser.compile<3>();

    const int N = 100;
    GpuArray<Real,3> dx{(hi[0]-lo[0]) / (N-1),
                        (hi[1]-lo[1]) / (N-1),
                        (hi[2]-lo[2]) / (N-1)};
    for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
    for (int k = 0; k < N; ++k) {
        Real x = lo[0] + i*dx[0];
        Real y = lo[1] + j*dx[1];
        Real z = lo[2] + k*dx[2];
        double result = exe(x,y,z);
        double benchmark = fb(x,y,z);
        double error = std::abs((result-benchmark)/(1.e-50+std::max(result,benchmark)));
        if (error > 1.e-15) {
            amrex::Print() << "f(" << x << "," << y << "," << z << ") = " << result << ", "
                           << benchmark << "\n";
        }
    }}}
}

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
//        Parser parser("a=3; a+4");
//        Parser parser("a=3; b=7; c=a+b; a+b+c");
//        Parser parser("a=x+3; b = x+y; b+a");
        Parser parser("max(0.1-abs(x+0.5), 0.1-abs(x-0.5))");
        parser.registerVariables({"x","y"});

        parser.print();
        amrex::Print() << "depth = " << parser.depth() << std::endl;
        auto const exe = parser.compile<2>();
        amrex::Print() << "max stack size = " << parser.maxStackSize() << std::endl;
        auto sym = parser.symbols();
        for (auto const& s: sym) {
            amrex::Print() << "Symbol: " << s << "\n";
        }

        amrex::Print() << "f() = " << exe({5,7}) << std::endl;

#if 0
        test3("a*sin(pi*x) + b*cos(2*pi*y+pi*z)",
              {{"a", 2.5}, {"b", -3.6}, {"pi", 3.1415926535897932}},
              {"x","y","z"},
              [=] (Real x, Real y, Real z) -> Real {
                  Real a = 2.5;
                  Real b = -3.6;
                  Real pi = 3.1415926535897932;
                  return a*sin(pi*x) + b*cos(2*pi*y+pi*z);
              },
              {0.0, 0.0, 0.0},
              {3.0, 3.0, 3.0});
#endif

//        test1("if(z<zp, nc*exp((z-zc)/lgrad), if(z<=zp2, 2.*nc, nc*exp(-(z-zc2)/lgrad)))",
//              {{"zp", 1.0}, {"nc", 1.e-6}, {"zc", 0.6}, {"lgrad", 2.5}, {"zp2", 2.0}, {"zc2", 1.3}},
//              {"z"});
//        test3("3.*2.*x*.4*0.5*sin(y)*2.*3.*z*4.*0.5",
//              {},
//              {"x","y","z"});
    }
    amrex::Print() << "\n";

    amrex::Finalize();
}
