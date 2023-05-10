#include <AMReX.H>
#include <AMReX_Gpu.H>

using namespace amrex;

template <typename T>
void f ()
{
    T t{};
    std::cout << t.name << "\n";
}

struct A {
    std::string name = "A";
};

struct B {
    std::string name = "B";
};

struct C {
    std::string name = "C";
};

template <int... Ops, typename... Ts>
void g (int type, amrex::CompileTimeOptions<Ops...>, amrex::TypeList<Ts...>)
{
    bool found = (false || ... || ((type == Ops) ? (f<Ts>(), true) : false));
    if (!found) {
        std::cout << "Unknown type " << type << "\n";
    }
}

int main(int argc, char* argv[])
{   
    amrex::Initialize(argc,argv,false);
    {
        int type = std::stoi(std::string(argv[1]));

        if (type == 10) {
            f<A>();
        } else if (type == 11) {
            f<B>();
        } else if (type == 12) {
            f<C>();            
        } else {
            std::cout << "Unknown type " << type << "\n";
        }

        // Same as the code above
        g(type,
          amrex::CompileTimeOptions<10,11,12>{},
          amrex::TypeList          <A ,B , C>{});
    }
    amrex::Finalize();
}
