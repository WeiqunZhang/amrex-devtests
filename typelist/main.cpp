#include <AMReX_Demangle.H>
#include <AMReX_TypeList.H>
#include <iostream>
#undef NDEBUG
#include <cassert>
using namespace amrex;

template<int ORD> void interp () { assert(0); }

template<> void interp<1> () { std::cout << "interp<1>\n"; }
template<> void interp<2> () { std::cout << "interp<2>\n"; }
template<> void interp<4> () { std::cout << "interp<4>\n"; }

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

    auto tp = TypeList<std::integral_constant<int,0>,
                       std::integral_constant<int,1>>{}
            + TypeList<std::integral_constant<int,2>,
                       std::integral_constant<int,3>>{};
    std::cout << "operator+: " << demangle(typeid(tp).name()) << "\n";

    auto cp = CartesianProduct(TypeList<std::integral_constant<int,0>,
                                        std::integral_constant<int,1>>{},
                               TypeList<std::integral_constant<int,2>,
                                        std::integral_constant<int,3>>{});
    std::cout << "cp: " << demangle(typeid(cp).name()) << "\n";

    int order = 2;
     ForEach(TypeList<std::integral_constant<int,1>,
                      std::integral_constant<int,2>,
                      std::integral_constant<int,4>>{},
             [&] (auto order_const) {
                 if (order_const() == order) {
                     interp<order_const()>();
                 }
             });
}
