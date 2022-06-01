#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

#include <memory>
#include <typeinfo>

using namespace amrex;

class Any
{
public:
    Any ()
        : m_ptr(std::make_unique<innards<int> >(0)) // store an int by default
        {}

    ~Any () = default;

    Any (Any const& rhs) = delete;  // We use this to store MultiFab etc. So let's delete copy ctor.
    Any& operator= (Any const& rhs) = delete;

    Any (Any && rhs) = default;
    Any& operator= (Any && rhs) = default;

    template <typename MF>
    Any (MF && mf)
        : m_ptr(std::make_unique<innards<MF> >(std::forward<MF>(mf)))
        {}

    template <typename MF>
    void operator= (MF && mf) {
        m_ptr = std::make_unique<innards<MF> >(std::forward<MF>(mf));
    }

    const std::type_info& Type () const {
        return m_ptr->Type();
    }

    template <typename MF>
    MF& get () { return dynamic_cast<innards<MF>&>(*m_ptr).m_mf; }

    template <typename MF>
    MF const& get () const { return dynamic_cast<innards<MF> const&>(*m_ptr).m_mf; }

private:
    struct innards_base {
        virtual const std::type_info& Type () const = 0;
    };

    template <typename MF>
    struct innards : innards_base
    {
        innards(MF && mf)
            : m_mf(std::forward<MF>(mf))
            {}

        virtual const std::type_info& Type () const override {
            return typeid(MF);
        }

        MF m_mf;
    };

    std::unique_ptr<innards_base> m_ptr;
};

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Any var;
        var = 42;
        amrex::Print() << "Type id: " << var.Type().name() << "\n"
                       << "    Is int? " << (var.Type() == typeid(int)) << "\n"
                       << "    value: " << var.get<int>() << std::endl;

        BoxArray ba(Box(IntVect(0),IntVect(63)));
        DistributionMapping dm(ba);
        var = MultiFab(ba, dm, 1, 0);
        var.get<MultiFab>().setVal(1.0);
        amrex::Print() << "Type id: " << var.Type().name() << " " << typeid(MultiFab).name() << "\n"
                       << "    Is MultiFab? " << (var.Type() == typeid(MultiFab)) << "\n"
                       << "    Sum: " << var.get<MultiFab>().sum() << std::endl;

        var = Array<MultiFab,AMREX_SPACEDIM>{};
        for (int i = 0; i < AMREX_SPACEDIM; ++i) {
            MultiFab& mf = var.get<Array<MultiFab,AMREX_SPACEDIM> >()[i];
            mf.define(amrex::convert(ba,IntVect::TheDimensionVector(i)), dm, 1, 0);
            mf.setVal(static_cast<Real>(i));
        }
        amrex::Print() << "Type id: " << var.Type().name() << "\n"
                       << "    Is Array<MultiFab,AMREX_SPACEDIM>()? " << (var.Type() == typeid(Array<MultiFab,AMREX_SPACEDIM>)) << "\n"
                       << "    Size: " << var.get<Array<MultiFab,AMREX_SPACEDIM> >().size() << std::endl;
        
    }
    amrex::Finalize();
}
