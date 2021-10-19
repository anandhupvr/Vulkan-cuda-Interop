#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>

#include <curand.h>
#include <curand_kernel.h>
#include <algorithm>
#define CUDA_DRIVER_API
#include <iostream>
#include "cudafuns.h"

int main() {

    typedef int ShareableHandle;
    ShareableHandle imgShareableHandle;

	cv::Mat img = cv::imread("/home/user1/Documents/works/Vulkan-cuda-Interop/statue.jpg", CV_LOAD_IMAGE_COLOR);
	CudaFuns cudafuns;
	
	imgShareableHandle = cudafuns.cudaops(img.data, img.cols, img.rows);
	cv::imshow("test", img);
	cv::waitKey(0);

	return 0;
}