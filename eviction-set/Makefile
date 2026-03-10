NVCC = nvcc
FLAGS = -arch=sm_89 -Xptxas -O0

FILES = finder.cu
BINS = $(FILES:.cu=)

all: $(BINS)

%: %.cu
	$(NVCC) $(FLAGS) $< -o $@

clean:
	rm -f $(BINS)

.PHONY: all clean

