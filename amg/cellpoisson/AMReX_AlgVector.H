#ifndef AMREX_ALG_VECTOR_H_
#define AMREX_ALG_VECTOR_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_FabArray.H>
#include <AMReX_INT.H>
#include <AMReX_LayoutData.H>

namespace amrex {

template <typename T, template<typename> typename Allocator = DefaultAllocator>
class AlgVector
{
public:
    using value_type = T;

    template <class U>
    using Vec = PODVector<U,Allocator<U> >;

    AlgVector (AlgPartition const& partition);

    ~AlgVector ();

    AlgVector (AlgVector const&) = delete;
    AlgVector (AlgVector &&) = delete;
    void operator= (AlgVector const&) = delete;
    void operator= (AlgVector &&) = delete;

    Long numLocalRows () const { return m_end - m_begin; }
    Long numGlobalRows () const { return m_partition.numRows(); }

    template <typename FAB,
              std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                               std::is_same<T,typename FAB::value_type>::value, int> = 0>
    void copyFrom (FabArray<FAB> const& fa);

    template <typename FAB,
              std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                               std::is_same<T,typename FAB::value_type>::value,int> = 0>
    void copyTo (FabArray<FAB> & fa) const;

private:
    AlgPartition m_partition;
    Long m_begin = 0;
    Long m_end = 0;
    Vec<T> m_data;
};

template <typename T, template<typename> typename Allocator>
AlgVector<T,Allocator>::AlgVector (AlgPartition const& partition)
    : m_partition(partition),
      m_begin(ParallelDescriptor::MyProc()),
      m_end(ParallelDescriptor::MyProc()+1),
      m_data(m_end-m_begin)
{}

template <typename T, template<typename> typename Allocator>
AlgVector<T,Allocator>::~AlgVector()
{}

template <typename T, template<typename> typename Allocator>
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                                         std::is_same<T,typename FAB::value_type>::value, int> >
void
AlgVector<T,Allocator>::copyFrom (FabArray<FAB> const& fa)
{
    AMREX_ASSERT(fa.is_cell_centered());

    LayoutData<T*> dptrs(fa.boxArray(), fa.DistributionMap());
    T* p = m_data.data();
    for (MFIter mfi(fa); mfi.isValid(); ++mfi) {
        dptrs[mfi] = p;
        p += mfi.validbox().numPts();
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
#pragma omp parallel
#endif
    for (MFIter mfi(fa); mfi.isValid(); ++mfi) {
        fa[mfi].template copyToMem<RunOn::Device>(mfi.validbox(), 0, 1, dptrs[mfi]);
    }
}

template <typename T, template<typename> typename Allocator>
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                                         std::is_same<T,typename FAB::value_type>::value, int> >
void
AlgVector<T,Allocator>::copyTo (FabArray<FAB> & fa) const
{
    AMREX_ASSERT(fa.is_cell_centered());

    LayoutData<T const*> dptrs(fa.boxArray(), fa.DistributionMap());
    T const* p = m_data.data();
    for (MFIter mfi(fa); mfi.isValid(); ++mfi) {
        dptrs[mfi] = p;
        p += mfi.validbox().numPts();
    }

#if defined(AMREX_USE_OMP) && !defined(AMREX_USE_GPU)
#pragma omp parallel
#endif
    for (MFIter mfi(fa); mfi.isValid(); ++mfi) {
        fa[mfi].template copyFromMem<RunOn::Device>(mfi.validbox(), 0, 1, dptrs[mfi]);
    }
}

}

#endif
