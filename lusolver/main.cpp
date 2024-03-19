#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_LUSolver.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    {
        LUSolver<6,Real> lusolver;
        Real alpha = 1.0;
        Real dx = 0.1;
        Real dy = 0.1;
        Real dz = 0.1;
        Real dxx = 1./(dx*dx);
        Real dyy = 1./(dy*dy);
        Real dzz = 1./(dz*dz);
        Real dxy = 1./(dx*dy);
        Real dxz = 1./(dx*dz);
        Real dyz = 1./(dy*dz);
        Real beta = alpha/(dx*dx) * 1.0e-3;

        Array2D<Real,0,5,0,5,Order::C> A
            {alpha*(dyy+dzz)*2.0 + beta,
             0.0,
             -alpha*dxy,
             alpha*dxy,
             -alpha*dxz,
             alpha*dxz,
             //
             0.0,
             alpha*(dyy+dzz)*2 + beta,
             alpha*dxy,
             -alpha*dxy,
             alpha*dxz,
             -alpha*dxz,
             //
             -alpha*dxy,
             alpha*dxy,
             alpha*(dxx+dzz)*2 + beta,
             0.0,
             -alpha*dyz,
             alpha*dyz,
             //
             alpha*dxy,
             -alpha*dxy,
             0.0,
             alpha*(dxx+dzz)*2 + beta,
             alpha*dyz,
             -alpha*dyz,
             //
             -alpha*dxz,
             alpha*dxz,
             -alpha*dyz,
             alpha*dyz,
             alpha*(dxx+dyy)*2 + beta,
             0.0,
             alpha*dxz,
             -alpha*dxz,
             alpha*dyz,
             -alpha*dyz,
             0.0,
             alpha*(dxx+dyy)*2 + beta};

        lusolver.define(A);

        Array<Real,6> b{1.0,-1.0,2.0,-2.0,3.0,-3.0};
        Array<Real,6> x;
        lusolver(x.data(), b.data());

        amrex::Print() << "A:\n";
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                std::cout << "   " << A(i,j) << ", ";
            }
            std::cout << std::endl;
        }

        amrex::Print() << "b:";
        for (int j = 0; j < 6; ++j) {
            amrex::Print() << "   " << b[j] << ", ";
        }
        amrex::Print() << std::endl;

        amrex::Print() << "Using LU, x:";
        for (int j = 0; j < 6; ++j) {
            amrex::Print().SetPrecision(17) << "   " << x[j] << ", ";
        }
        amrex::Print() << std::endl;

        amrex::Print() << "Using LU, A*x: ";
        for (int i = 0; i < 6; ++i) {
            Real r = 0.0;
            for (int j = 0; j < 6; ++j) {
                r += A(i,j) * x[j];
            }
            amrex::Print().SetPrecision(17) << "   " << r << ", ";
        }
        amrex::Print() <<  std::endl;

        auto IA = lusolver.invert();
        amrex::Print() << "A^-1:\n";
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                std::cout << "   " << IA(i,j) << ", ";
            }
            std::cout << std::endl;
        }

        Array<Real,6> x2;
        for (int i = 0; i < 6; ++i) {
            Real r = 0.0;
            for (int j = 0; j < 6; ++j) {
                r += IA(i,j) * b[j];
            }
            x2[i] = r;
        }

        amrex::Print() << "Using A^-1, x:";
        for (int j = 0; j < 6; ++j) {
            amrex::Print().SetPrecision(17) << "   " << x2[j] << ", ";
        }
        amrex::Print() << std::endl;

        amrex::Print() << "Using A^-1, A*x: ";
        for (int i = 0; i < 6; ++i) {
            Real r = 0.0;
            for (int j = 0; j < 6; ++j) {
                r += A(i,j) * x2[j];
            }
            amrex::Print().SetPrecision(17) << "   " << r << ", ";
        }
        amrex::Print() <<  std::endl;

        amrex::Print() << "Determinant of A: " << lusolver.determinant() << std::endl;
    }
    amrex::Finalize();
}
