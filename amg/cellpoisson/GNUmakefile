AMREX_HOME ?= ../../../amrex

DEBUG	= FALSE

DIM	= 2

COMP    = gcc

USE_CUDA  = TRUE
USE_HIP   = FALSE
USE_DPCPP = FALSE

USE_MPI   = TRUE
USE_OMP   = FALSE

BL_NO_FORT = TRUE

include $(AMREX_HOME)/Tools/GNUMake/Make.defs

ifeq ($(USE_CUDA),TRUE)
  LIBRARIES += -lcusparse
endif

ifeq ($(USE_HIP),TRUE)
  LIBRARY_LOCATIONS += $(ROC_PATH)/rocsparse/lib
  LIBRARIES += -Wl,--rpath=$(ROC_PATH)/rocsparse/lib -lrocsparse
endif

include ./Make.package

include ../AMG/Make.package
VPATH_LOCATIONS += ../AMG
INCLUDE_LOCATIONS += ../AMG

include $(AMREX_HOME)/Src/Base/Make.package

include $(AMREX_HOME)/Tools/GNUMake/Make.rules
