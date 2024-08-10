#pragma once

#include <AMReX_Gpu.H>

namespace oneapiexp = sycl::ext::oneapi::experimental;

SYCL_EXTERNAL extern oneapiexp::device_global<amrex::Real> dg_x;
SYCL_EXTERNAL extern oneapiexp::device_global<amrex::Real[4]> dg_y;
