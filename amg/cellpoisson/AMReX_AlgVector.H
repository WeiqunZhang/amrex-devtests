#ifndef AMREX_ALG_VECTOR_H_
#define AMREX_ALG_VECTOR_H_
#include <AMReX_Config.H>

#include <AMReX_AlgPartition.H>
#include <AMReX_FabArray.H>
#include <AMReX_INT.H>
#include <AMReX_LayoutData.H>

#include <fstream>
#include <string>
#include <type_traits>

namespace amrex {

template <typename T, typename Allocator = DefaultAllocator<T> >
class AlgVector
{
public:
    using value_type = T;
    using allocator_type = Allocator;

    using Vec = PODVector<T,Allocator>;

    AlgVector (AlgPartition const& partition);

    AlgVector (AlgVector<T, Allocator> const&) = delete;
    AlgVector& operator= (AlgVector<T, Allocator> const&) = delete;

    AlgVector (AlgVector<T, Allocator> &&) noexcept = default;
    AlgVector& operator= (AlgVector<T, Allocator> &&) noexcept = default;

    AlgPartition const& partition () const { return m_partition; }

    Long numLocalRows () const { return m_end - m_begin; }
    Long numGlobalRows () const { return m_partition.numGlobalRows(); }

    T const* data () const { return m_data.data(); }
    T      * data ()       { return m_data.data(); }

    void setVal (T val);

    template <typename FAB,
              std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                               std::is_same<T,typename FAB::value_type>::value, int> = 0>
    void copyFrom (FabArray<FAB> const& fa);

    template <typename FAB,
              std::enable_if_t<amrex::IsBaseFab<FAB>::value &&
                               std::is_same<T,typename FAB::value_type>::value,int> = 0>
    void copyTo (FabArray<FAB> & fa) const;

    void printToFile (std::string const& file) const;

private:
    AlgPartition m_partition;
    Long m_begin = 0;
    Long m_end = 0;
    Vec m_data;
};

template <typename T, typename Allocator>
AlgVector<T,Allocator>::AlgVector (AlgPartition const& partition)
    : m_partition(partition),
      m_begin(partition[ParallelDescriptor::MyProc()]),
      m_end(partition[ParallelDescriptor::MyProc()+1]),
      m_data(m_end-m_begin)
{
    static_assert(std::is_floating_point<T>::value, "AlgVector is for floating point type only");
}

template <typename T, typename Allocator>
void
AlgVector<T,Allocator>::setVal (T val)
{
    Long n = m_data.size();
    T* p = m_data.data();
    ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) noexcept { p[i] = val; });
    Gpu::streamSynchronize();
}

template <typename T, typename Allocator>
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

template <typename T, typename Allocator>
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

template <typename T, typename Allocator>
void AlgVector<T,Allocator>::printToFile (std::string const& file) const
{
    std::ofstream ofs(file+"."+std::to_string(ParallelDescriptor::MyProc()));
    AllPrintToFile os(file);
    ofs << m_begin << " " << m_end << "\n";
    for (Long i = 0, N = m_data.size(); i < N; ++i) {
        ofs << i+m_begin << " " << m_data[i] << "\n";
    }
}

template <class V, class Enable = void> struct IsAlgVector : std::false_type {};
//
template <class V>
struct IsAlgVector<V, typename std::enable_if<std::is_same<AlgVector<typename V::value_type,
                                                                     typename V::allocator_type>,
                                                           V>::value>::type>
    : std::true_type {};

template <typename V1, typename V2, typename F>
std::enable_if_t<IsAlgVector<std::decay_t<V1> >::value &&
                 IsAlgVector<std::decay_t<V2> >::value>
ForEach (V1 && x, V2 && y, F&& f)
{
    AMREX_ASSERT(x.numLocalRows() == y.numLocalRows());
    Long n = x.numLocalRows();
    auto px = x.data();
    auto py = y.data();
    ParallelFor(n, [=] AMREX_GPU_DEVICE (Long i) noexcept
    {
        f(px[i], py[i]);
    });
}

}

#endif
