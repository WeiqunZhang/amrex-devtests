#include <AMReX.H>

#include <iostream>
#include <cstdlib>
#include <sstream>

using namespace amrex;

namespace {

void print_usage ()
{
    std::cout << "\n This MPI program splits its communicator into two, one for"
              << "\n GPU and the other for CPU."
              << "\n"
              << "\n Example usage:"
              << "\n    mpiexec -n 6 ./main3d.gnu.MPI.ex [inputs] -- 2\n"
              << "\n Here '2' is the number of processes for the GPU part of the"
              << "\n partition.\n\n";
}

}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int myproc, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &myproc);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    int dashdash = 0;
    for (; dashdash < argc; ++dashdash) {
        if (std::strcmp(argv[dashdash], "--") == 0) { break; }
    }

    int ngpus = 0;
    if (dashdash < argc-1) {
        std::istringstream iss(argv[argc-1]);
        iss >> ngpus;
    } else {
        if (myproc == 0) {
            print_usage();
        }
        return EXIT_FAILURE;
    }

    std::cout << "ngpus = " << ngpus << "\n";

    MPI_Finalize();
}
