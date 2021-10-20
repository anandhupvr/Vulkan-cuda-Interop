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

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "cudafuns.h"
#include "VulkanBase.h"




int main() {

    typedef int ShareableHandle;
    ShareableHandle imgShareableHandle;

	cv::Mat img = cv::imread("/home/user1/Documents/works/Vulkan-cuda-Interop/statue.jpg", CV_LOAD_IMAGE_COLOR);
	CudaFuns cudafuns;


    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();

    const int bufSize = width/2 * height/2 * 4;
    
    //the handle obtained here will be passed to vulkan to import the allocation
    imgShareableHandle = cudafuns.cudaops((unsigned char*) img.data, width, height, width/2, height/2);

    VulkanBase app(width/2, height);
    try {
        app.run((void *)(uintptr_t)imgShareableHandle, app.getDefaultMemHandleType(), bufSize * sizeof(unsigned char));
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

	return 0;
}