#include <AMReX_AlgVector.H>

namespace amrex {

template <typename T>
AlgVector<T>::AlgVector<T> (Long a_row_begin, Long a_row_end, Arena* ar)
    : DataAllocator(ar),
      m_row_begin(a_row_begin),
      m_row_end(a_row_end)
{
    // allocate memory
}

template <typename T>
AlgVector<T>::~AlgVector<T> ()
{
    // delete memory
}

}
