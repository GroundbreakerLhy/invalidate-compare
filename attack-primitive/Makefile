all: delta victim llm_victim disruptor

delta: primitive-example.cu eviction-sets.h
	nvcc -arch=sm_89 -Xptxas -O0 primitive-example.cu -o delta

victim: victim.cu
	nvcc -arch=sm_89 victim.cu -o victim

llm_victim: llm_victim.cu
	nvcc -arch=sm_89 llm_victim.cu -o llm_victim

disruptor: disruptor.cu
	nvcc -arch=sm_89 disruptor.cu -o disruptor

llm_attack: delta llm_victim disruptor
