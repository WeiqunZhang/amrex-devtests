#include <AMReX_FFT_Poisson.H> // Put this at the top for testing

#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace amrex;

namespace {

struct SolveConfig {
    std::string label;
    bool padding = true;
    int nfactors = 3;
};

struct SolveResult {
    std::string label;
    IntVect cells;
    IntVect original_size;
    IntVect padded_size;
    double wall_time = 0.0;
};

std::string
to_string (IntVect const& iv)
{
    std::ostringstream os;
    os << iv;
    return os.str();
}

using TableRow = std::array<std::string,5>;

void
print_org_row (std::ostringstream& os, TableRow const& row,
               std::array<std::size_t,5> const& widths)
{
    os << "|";
    for (std::size_t i = 0; i < row.size(); ++i) {
        os << " " << std::left << std::setw(static_cast<int>(widths[i])) << row[i] << " |";
    }
    os << "\n";
}

void
print_org_separator (std::ostringstream& os, std::array<std::size_t,5> const& widths)
{
    os << "|";
    for (std::size_t i = 0; i < widths.size(); ++i) {
        os << std::string(widths[i]+2, '-');
        os << ((i+1 == widths.size()) ? "|" : "+");
    }
    os << "\n";
}

void
fill_rhs (MultiFab& rho, Geometry const& geom)
{
    auto const dx = geom.CellSizeArray();
    auto const problo = geom.ProbLoArray();
    auto const& rho_arrays = rho.arrays();

    ParallelFor(rho, [=] AMREX_GPU_DEVICE (int b, int i, int j, int k) noexcept
    {
        Real const x = problo[0] + Real(i)*dx[0];
        Real const y = problo[1] + Real(j)*dx[1];
        Real const z = problo[2] + Real(k)*dx[2];
        rho_arrays[b](i,j,k) = std::exp(-12._rt*(x*x + y*y + z*z));
    });
}

FFT::Info
make_info (SolveConfig const& config)
{
    FFT::Info info;
    info.setOpenBCPadding(config.padding);
    if (config.padding) {
        info.setOpenBCPaddingFactors(config.nfactors);
    }
    return info;
}

SolveResult
run_solves (SolveConfig const& config, Geometry const& geom,
            MultiFab& phi, MultiFab const& rho, IntVect const& cells,
            IntVect const& original_size, int nsolve)
{
    FFT::PoissonOpenBC solver(geom, IndexType::TheNodeType(), IntVect(1),
                              make_info(config));

    Gpu::synchronize();
    solver.solve(phi, rho);
    Gpu::synchronize();

    std::string const profile_name = std::string("openbc_fft_") + config.label;
    double wall_time = 0.0;
    {
        BL_PROFILE_REGION(profile_name);
        BL_PROFILE(profile_name);
        Gpu::synchronize();
        double const t0 = amrex::second();
        for (int isolve = 0; isolve < nsolve; ++isolve) {
            solver.solve(phi, rho);
        }
        Gpu::synchronize();
        wall_time = amrex::second() - t0;
    }

    ParallelDescriptor::ReduceRealMax(wall_time, ParallelDescriptor::IOProcessorNumber());

    return SolveResult{config.label, cells, original_size, solver.PaddedLength(), wall_time};
}

void
print_summary (Vector<SolveResult> const& results)
{
    if (!ParallelDescriptor::IOProcessor()) {
        return;
    }

    std::ostringstream os;
    os << "\nOpenBC FFT padding solve-time summary\n";
    TableRow const header{{"solver", "cells", "original", "padded", "wall time (s)"}};
    std::array<std::size_t,5> widths{};
    for (std::size_t i = 0; i < header.size(); ++i) {
        widths[i] = header[i].size();
    }

    Vector<TableRow> rows;
    rows.reserve(results.size());
    for (auto const& result : results) {
        std::ostringstream time;
        time << std::fixed << std::setprecision(6) << result.wall_time;
        rows.push_back(TableRow{{result.label, to_string(result.cells),
                                 to_string(result.original_size),
                                 to_string(result.padded_size), time.str()}});
        for (std::size_t i = 0; i < rows.back().size(); ++i) {
            widths[i] = std::max(widths[i], rows.back()[i].size());
        }
    }

    print_org_row(os, header, widths);
    print_org_separator(os, widths);
    for (auto const& row : rows) {
        print_org_row(os, row, widths);
    }

    amrex::Print() << os.str();
}

}

int
main (int argc, char* argv[])
{
    static_assert(AMREX_SPACEDIM == 3);

    amrex::Initialize(argc, argv);
    {
        BL_PROFILE("main");

        int n_cell = 64;
        int n_cell_x = n_cell;
        int n_cell_y = n_cell;
        int n_cell_z = n_cell;

        int max_grid_size_x = 32;
        int max_grid_size_y = 32;
        int max_grid_size_z = 32;

        int nsolve = 10;

        ParmParse pp;
        pp.query("n_cell", n_cell);
        n_cell_x = n_cell;
        n_cell_y = n_cell;
        n_cell_z = n_cell;
        pp.query("n_cell_x", n_cell_x);
        pp.query("n_cell_y", n_cell_y);
        pp.query("n_cell_z", n_cell_z);
        pp.query("max_grid_size_x", max_grid_size_x);
        pp.query("max_grid_size_y", max_grid_size_y);
        pp.query("max_grid_size_z", max_grid_size_z);
        pp.query("nsolve", nsolve);

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            n_cell_x > 0 && n_cell_y > 0 && n_cell_z > 0,
            "n_cell_x, n_cell_y and n_cell_z must be positive");
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nsolve > 0, "nsolve must be positive");

        Box const domain(IntVect(0), IntVect(n_cell_x-1, n_cell_y-1, n_cell_z-1));
        BoxArray ba(domain);
        ba.maxSize(IntVect(max_grid_size_x, max_grid_size_y, max_grid_size_z));
        DistributionMapping dm(ba);

        Geometry geom(domain,
                      RealBox(-1._rt, -1._rt, -1._rt, 1._rt, 1._rt, 1._rt),
                      CoordSys::cartesian, {0, 0, 0});

        BoxArray nodal_ba = amrex::convert(ba, IndexType::TheNodeType());
        MultiFab rho(nodal_ba, dm, 1, 0);
        MultiFab phi(nodal_ba, dm, 1, 1);

        fill_rhs(rho, geom);
        phi.setVal(0._rt);
        Gpu::synchronize();

        Box const grown_nodal_domain = amrex::grow(
            amrex::convert(domain, IndexType::TheNodeType()), IntVect(1));
        IntVect const original_size = grown_nodal_domain.length();

        IntVect const cells(n_cell_x, n_cell_y, n_cell_z);

        amrex::Print() << "OpenBC FFT padding performance test\n"
                       << "  cells: " << cells << "\n"
                       << "  timed solves per configuration: " << nsolve << "\n";

        std::array<SolveConfig,4> const configs{{
            {"unpadded", false, 3},
            {"nfactors=3", true, 3},
            {"nfactors=4", true, 4},
            {"nfactors=5", true, 5}
        }};

        Vector<SolveResult> results;
        results.reserve(configs.size());
        for (auto const& config : configs) {
            results.push_back(run_solves(config, geom, phi, rho, cells, original_size, nsolve));
        }

        print_summary(results);
    }
    amrex::Finalize();
}
