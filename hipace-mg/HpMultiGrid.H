#ifndef HP_MULTIGRID_H_
#define HP_MULTIGRID_H_

#include <AMReX_FArrayBox.H>

namespace hpmg {

// Solve `-acoef * sol + Lap(sol) = rhs` with homogeneous Dirichlet BC.
// sol and rhs have two components, whereas acoef has only one component.
class MultiGrid
{
public:

    explicit MultiGrid (amrex::Box const& a_domain);

    void solve (amrex::FArrayBox& sol, amrex::FArrayBox const& rhs,
                amrex::FArrayBox const& acoef, amrex::Real dx, amrex::Real dy,
                amrex::Real tol_rel, amrex::Real tol_abs, int nummaxiter, int verbose);

// public for cuda
    void average_down_acoef ();
    void vcycle ();
    void bottomsolve ();

private:

    amrex::Vector<amrex::Box> m_domain;
    amrex::Real m_dx, m_dy;

    int m_max_level;
    int m_single_block_level_begin;
    int m_num_mg_levels;
    int m_num_single_block_levels;

    amrex::FArrayBox m_sol;
    amrex::FArrayBox m_rhs;

    static constexpr int nfabvs = 4;
    amrex::Vector<amrex::FArrayBox> m_acf;
    amrex::Vector<amrex::FArrayBox> m_res;
    amrex::Vector<amrex::FArrayBox> m_cor;
    amrex::Vector<amrex::FArrayBox> m_rescor;

    amrex::Array4<amrex::Real> const* m_acf_a = nullptr;
    amrex::Array4<amrex::Real> const* m_res_a = nullptr;
    amrex::Array4<amrex::Real> const* m_cor_a = nullptr;
    amrex::Array4<amrex::Real> const* m_rescor_a = nullptr;

    amrex::Gpu::PinnedVector<amrex::Array4<amrex::Real> > m_h_array4;
    amrex::Gpu::DeviceVector<amrex::Array4<amrex::Real> > m_d_array4;
};

}

#endif
