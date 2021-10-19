#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#define CUDA_DRIVER_API



class CudaFuns {

public:
	CudaFuns(){};
	void allocateMem(size_t imgSize);
	int cudaops(unsigned char* input, int width, int height);
	// CU_MEM_HANDLE_TYPE_WIN32 for windows
	// CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR for linux
	CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	CUdeviceptr d_ptr = 0U;
    typedef int ShareableHandle;
    ShareableHandle imgShareableHandle;

};