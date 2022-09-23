CXX = nvcc

ifeq ($(CXX),nvcc)
  CXXFLAGS = -ccbin=g++ -Xcompiler='-O3 -std=c++17' -O3 --std=c++17 -m64 -arch=compute_70 -code=sm_70 --expt-extended-lambda -x cu -DAMREX_USE_CUDA
else ifeq ($(CXX),nvc++)
  CXXFLAGS = -O3 --std=c++17 -cuda -gpu=cc70 -DAMREX_USE_CUDA
else ifeq ($(CXX),hipcc)
  CXXFLAGS = -O3 --std=c++17 -m64 --amdgpu-target=gfx90a -DAMREX_USE_HIP
else ifeq ($(CXX),dpcpp)
  CXXFLAGS = -O3 --std=c++17 -fsycl -DAMREX_USE_DPCPP
else
  $(error Unknown CXX)
endif

a.out: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	${RM} *.o a.out

FORCE:

.PHONY: clean
