#define AMREX_USE_CXXABI
#include <AMReX_TypeList.H>
#include <iostream>
#undef NDEBUG
#include <cassert>
using namespace amrex;
int main(int argc, char* argv[])
{
    auto tt = CartesianProduct(TypeList<char,int,long>{},
                               TypeList<float,double>{});
    static_assert(tt.size() == 6);

    ForEach(tt, [] (auto t) {
        std::cout << "    " << demangle(typeid(t).name()) << "\n";
    });

    std::cout << "\n";

    bool r = ForEachUntil(tt, [] (auto t) -> bool {
        if (std::is_same<TypeAt<0,decltype(t)>, long>::value &&
            std::is_same<TypeAt<1,decltype(t)>, float>::value) {
            std::cout << "    Found <long,float>\n";
            return 1;
        } else {
            std::cout << "    This is " << demangle(typeid(t).name()) << "\n";
            return 0;
        }
    });
    assert(r);

    r = ForEachUntil(tt, [] (auto t) -> bool {
        if (std::is_same<TypeAt<0,decltype(t)>, unsigned>::value &&
            std::is_same<TypeAt<1,decltype(t)>, float>::value) {
            std::cout << "    Found <unsigned,float>\n";
            return 1;
        } else {
            return 0;
        }
    });
    assert(!r);
}
