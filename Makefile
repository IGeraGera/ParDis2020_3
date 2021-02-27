CC=nvcc


main:
	$(CC) main.cu iofun.cu imgfun.cu gaussian.cu -o main

test:
	./main data/test.csv 3

clean:
	-rm -f main
out_clean:
	-rm data/test_out*

