#ifndef FOO_H_
#define FOO_H_

#include <AMReX_GpuQualifiers.H>

extern AMREX_GPU_DEVICE int d_a;
extern AMREX_GPU_DEVICE int d_b[4];

extern AMREX_GPU_MANAGED int m_a;
extern AMREX_GPU_MANAGED int m_b[4];

#endif
