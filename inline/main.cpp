#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Print.H>

using namespace amrex;

AMREX_FORCE_INLINE
void test1 (int i, int j, int k, Array4<Real> const& a, Array4<Real const> const& b, Array4<Real const> const& c)
{
    a(i,j,k) = b(i,j,k) + 3. * c(i,j,k);
}

void test2 (int i, int j, int k, Array4<Real> const& a, Array4<Real const> const& b, Array4<Real const> const& c);

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        Box box(IntVect(0),IntVect(63));
        BoxArray ba(box);
        ba.maxSize(32);
        DistributionMapping dm{ba};
        MultiFab mfa(ba,dm,1,0);
        MultiFab mfb(ba,dm,1,0);
        MultiFab mfc(ba,dm,1,0);
        mfa.setVal(0.0);
        mfb.setVal(1.0);
        mfc.setVal(2.0);

        double t0 = amrex::second();
        for (MFIter mfi(mfa,true); mfi.isValid(); ++mfi) {
            Array4<Real> const& a = mfa.array(mfi);
            Array4<Real const> const& b = mfb.const_array(mfi);
            Array4<Real const> const& c = mfc.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] (int i, int j, int k)
            {
                test1(i,j,k,a,b,c);
            });
        }
        double t1 = amrex::second();
        for (MFIter mfi(mfa,true); mfi.isValid(); ++mfi) {
            Array4<Real> const& a = mfa.array(mfi);
            Array4<Real const> const& b = mfb.const_array(mfi);
            Array4<Real const> const& c = mfc.const_array(mfi);
            amrex::ParallelFor(mfi.tilebox(), [=] (int i, int j, int k)
            {
                test2(i,j,k,a,b,c);
            });
        }
        double t2 = amrex::second();
        amrex::Print() << "inline test1 time is " << t1-t0 << ", no inline test2 time is " << t2-t1 << ", the ratio is " << (t2-t1)/(t1-t0) << std::endl;
    }
    amrex::Finalize();
}
