#include "cudafuns.h"

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)

void getDefaultSecurityDescriptor(CUmemAllocationProp *prop) {
#if defined(__linux__)
    return;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
    static OBJECT_ATTRIBUTES objAttributes;
    static bool objAttributesConfigured = false;

    if (!objAttributesConfigured) {
        PSECURITY_DESCRIPTOR secDesc;
        BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
            sddl, SDDL_REVISION_1, &secDesc, NULL);
        if (result == 0) {
            printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n",
                GetLastError());
        }

        InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);

        objAttributesConfigured = true;
    }

    prop->win32HandleMetaData = &objAttributes;
    return;
#endif
}

__global__ void copy_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    //2D Index of current thread
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    int channel =4;
    //Only valid threads perform memory I/O
    if((x<width) && (y<height))
    {   
        // converting bgr to rgba format for vulkan
        output[((y * width * channel) + x * channel) + 0 ] = input[((y * width * 3) + x * 3) + 2 ];
        output[((y * width * channel) + x * channel) + 1 ] = input[((y * width * 3) + x * 3) + 1 ];
        output[((y * width * channel) + x * channel) + 2 ] = input[((y * width * 3) + x * 3) + 0 ];

        output[((y * width * channel) + x * channel) + 3 ] = (unsigned char)0;
    }
}

void CudaFuns::allocateMem(size_t imgSize){

    int cudaDevice = 0;
    size_t granularity = 0;
    cudaSetDevice(cudaDevice);

    CUmemGenericAllocationHandle cudaImgHandle;


    CUmemAllocationProp allocProp = { };
    allocProp.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocProp.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocProp.location.id = cudaDevice;
    allocProp.win32HandleMetaData = NULL;
    allocProp.requestedHandleTypes = ipcHandleTypeFlag;

    getDefaultSecurityDescriptor(&allocProp);

    cuMemGetAllocationGranularity(&granularity, &allocProp, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);

    size_t sizeRounded = ROUND_UP_TO_GRANULARITY(imgSize, granularity);
    cuMemAddressReserve(&d_ptr, sizeRounded, granularity, 0U, 0);
    cuMemCreate(&cudaImgHandle, sizeRounded, &allocProp, 0);

    // Shareable Handles(a file descriptor on Linux and NT Handle on Windows), used for sharing cuda
    // allocated memory with Vulkan
    cuMemExportToShareableHandle((void *)&imgShareableHandle, cudaImgHandle, ipcHandleTypeFlag, 0);


    cuMemMap(d_ptr, imgSize, 0, cudaImgHandle, 0);
    cuMemRelease(cudaImgHandle);

    CUmemAccessDesc accessDescriptor = {};
    accessDescriptor.location.id = cudaDevice;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Apply the access descriptor to the whole VA range. Essentially enables Read-Write access to the range.
    cuMemSetAccess(d_ptr, imgSize, &accessDescriptor, 1);


}

int CudaFuns::cudaops(unsigned char* input, int width, int height) {

    size_t imgSize = width * height * 4;

    allocateMem(imgSize);

    // Pointer to Cuda allocated buffers which are imported and used by vulkan as vertex buffer
    unsigned char *d_output;

    d_output = (unsigned char*)d_ptr;
    size_t colorBytes = width * height * 3;
    unsigned char *d_input;
    cudaMalloc<unsigned char>(&d_input, colorBytes);
    cudaMemcpy(d_input, input, colorBytes, cudaMemcpyHostToDevice);
    const dim3 block(16,16);
    const dim3 grid((width + block.x - 1)/block.x, (height + block.y - 1)/block.y);

    copy_kernel<<<grid,block>>>(d_input, d_output, width, height);
    cudaDeviceSynchronize();
    cudaFree(d_input);

    return imgShareableHandle;

}

