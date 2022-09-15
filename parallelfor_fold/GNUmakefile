USE_CUDA = FALSE
USE_HIP = FALSE

CXX = g++
CXXFLAGS = -g -O3 -std=c++17

a.out: main.o
	$(CXX) -o $@ $^

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	${RM} *.o a.out

FORCE:

.PHONY: clean
