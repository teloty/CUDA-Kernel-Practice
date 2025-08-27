nvcc 00_hello_world.cu -o hello
./hello
rm -rf ./hello

nvcc 01_reduce_sum.cu -o reducesum
./reducesum
rm -rf ./reducesum

nvcc 02_gemm.cu -o gemm -lcublas
./gemm
rm -rf ./gemm
