USE_CUDA=TRUE

ifeq ($(USE_CUDA),TRUE)

  CXX = nvcc
  CXXFLAGS = -ccbin=g++ -Xcompiler='-g -O3 -std=c++17' --std=c++17 -m64 -arch=compute_70 -code=sm_70 --expt-extended-lambda -dc -x cu
  LINKFLAGS = -arch=compute_70 -code=sm_70

else

  CXX = hipcc
  CXXFLAGS = -std=c++17 -m64 -g -O3 --amdgpu-target=gfx90a -fgpu-rdc -Wno-unused-result
  LINKFLAGS = --amdgpu-target=gfx90a -fgpu-rdc

endif

a.out: main.o
	$(CXX) $(LINKFLAGS) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	${RM} *.o a.out

FORCE:

.PHONY: clean
