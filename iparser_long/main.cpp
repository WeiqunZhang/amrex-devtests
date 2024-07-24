#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_IParser.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    {
        ParmParse pp("amrex");
        pp.add("signal_handling", 0);
        pp.add("throw_exception", 1);
    }
    amrex::Initialize(argc, argv);
    {
        std::vector<std::string> vs{"1'234'567'890'123", // good
                                    "-1000",             // good
                                    "5e2",               // good
                                    "5.400e4",           // good
                                    "-5.4e2",            // good
                                    ".123e5",            // good
                                    "0.123e5",           // good
                                    "5.43e2",            // good
                                    "1234e0",            // good         
                                    "1234.567e3",        // good
                                    "3e-4",              // bad
                                    "5.432e2",           // bad
                                    "10000e-3",          // bad
                                    "3.14"};             // bad
        for (auto const& s : vs) {
            std::cout << "s = " << std::setw(18) << s;
            try {
                IParser iparser(s);
                auto exe = iparser.compile<0>();
                auto i = exe();
                std::cout << "    i = " << i << "\n";
            } catch (std::runtime_error const& e) {
                std::cout << "    error: " << e.what() << '\n';
            }
        }
    }

    amrex::Finalize();
}
