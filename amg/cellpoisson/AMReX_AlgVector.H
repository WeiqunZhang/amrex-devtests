#ifndef AMREX_ALG_VECTOR_H_
#define AMREX_ALG_VECTOR_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_FabArray.H>
#include <AMReX_INT.H>
#include <AMReX_LayoutData.H>

#include <memory>

namespace amrex {

template <typename T>
class AlgVector
    : public DataAllocator
{
public:
    typedef T value_type;

    AlgVector (AlgPartition const& partition, Arena* ar = nullptr);

    ~AlgVector ();

    AlgVector (AlgVector const&) = delete;
    void operator= (AlgVector const&) = delete;

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
    Long m_begin, m_end;
    std::unique_ptr<T,DataDeleter> m_data;
};

template <typename T>
AlgVector<T>::AlgVector (AlgPartition const& partition, Arena* ar)
    : DataAllocator(ar),
      m_partition(partition),
      m_begin(ParallelDescriptor::MyProc()),
      m_end(ParallelDescriptor::MyProc()+1)
{
    if (m_end > m_begin) {
        m_data = std::unique_ptr<T,DataDeleter>((T*)(this->alloc(sizeof(T)*(m_end-m_begin))),
                                                DataDeleter{this->arena()});
    }
}

template <typename T>
AlgVector<T>::~AlgVector()
{
}

template <typename T>
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                                         std::is_same<T,typename FAB::value_type>::value, int> >
void
AlgVector<T>::copyFrom (FabArray<FAB> const& fa)
{
    AMREX_ASSERT(fa.is_cell_centered());

    LayoutData<T*> dptrs(fa.boxArray(), fa.DistributionMap());
    T* p = m_data.get();
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

template <typename T>
template <typename FAB, std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                                         std::is_same<T,typename FAB::value_type>::value, int> >
void
AlgVector<T>::copyTo (FabArray<FAB> & fa) const
{
    AMREX_ASSERT(fa.is_cell_centered());

    LayoutData<T const*> dptrs(fa.boxArray(), fa.DistributionMap());
    T const* p = m_data.get();
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
