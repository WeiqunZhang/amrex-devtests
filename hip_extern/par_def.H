#ifndef PAR_DEF_H_
#define PAR_DEF_H_

#include "par.H"

AMREX_GPU_DEVICE int d_a;
AMREX_GPU_DEVICE int d_b[4];

AMREX_GPU_MANAGED int m_a;
AMREX_GPU_MANAGED int m_b[4];

#endif
