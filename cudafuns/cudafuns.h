#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#define CUDA_DRIVER_API



class CudaFuns {

public:
	CudaFuns(){};
	void allocateMem();
	void justCopy(const cv::Mat& input, unsigned char* output);
};