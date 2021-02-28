CC=nvcc


main:
	$(CC) main.cu iofun.cu imgfun.cu gaussian.cu -o main

mainMemoryLimit:
	$(CC) mainMemoryLimit.cu iofun.cu imgfun.cu gaussian.cu -o mainMemoryLimit
test:
	./main image64.csv 3

clean:
	-rm -f main
	-rm -f mainMemoryLimit
	-rm image64_out*

