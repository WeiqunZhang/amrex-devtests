AMREX_HOME := ../../amrex

DEBUG	= FALSE

DIM	= 3

COMP    = gcc

USE_MPI   = TRUE
USE_OMP   = FALSE
USE_CUDA  = FALSE
USE_HIP   = FALSE
USE_SYCL  = FALSE

BL_NO_FORT = TRUE

TINY_PROFILE = FALSE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

include ./Make.package
include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/FFT/Make.package

ifeq ($(USE_HEFFTE),TRUE)
  USERSuffix += .HEFFTE
  DEFINES += -DUSE_HEFFTE
  HEFFTE_HOME ?= $(HOME)/opt/heffte-fftw
  VPATH_LOCATIONS += $(HEFFTE_HOME)/include
  INCLUDE_LOCATIONS += $(HEFFTE_HOME)/include
  LIBRARY_LOCATIONS += $(HEFFTE_HOME)/lib
  libraries += -lheffte
ifneq ($(USE_GPU),TRUE)
  libraries += -lfftw3_mpi
endif
endif

ifeq ($(USE_CUDA),TRUE)
  libraries += -lcufft
else ifeq ($(USE_HIP),TRUE)
  # Use rocFFT.  ROC_PATH is defined in amrex
  INCLUDE_LOCATIONS += $(ROC_PATH)/rocfft/include
  LIBRARY_LOCATIONS += $(ROC_PATH)/rocfft/lib
  LIBRARIES += -L$(ROC_PATH)/rocfft/lib -lrocfft
else
  libraries += -lfftw3
endif

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
