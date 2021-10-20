#include "cudafuns.h"

#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)
CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
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


void CudaFuns::allocateMem(size_t outSize){

    int deviceCount;
    int cudaDevice = cudaInvalidDeviceId;
    cudaGetDeviceCount(&deviceCount);
    // need to fix
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp devProp = { };
        cudaGetDeviceProperties(&devProp, dev);
        if (true) {
            cudaDevice = dev;
            break;
        }

        // if (isVkPhysicalDeviceUuid(&devProp.uuid)) {
        //     cudaDevice = dev;
        //     break;
        // }
    }
    // if (cudaDevice == cudaInvalidDeviceId) {
    //     throw std::runtime_error("No Suitable device found!");
    // }
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
    size_t imgSize = ROUND_UP_TO_GRANULARITY(outSize, granularity);
    cuMemAddressReserve(&d_ptr, imgSize, granularity, 0U, 0);
    cuMemCreate(&cudaImgHandle, imgSize, &allocProp, 0);
    cuMemExportToShareableHandle((void *)&imgShareableHandle, cudaImgHandle, ipcHandleTypeFlag, 0);



    cuMemMap(d_ptr, imgSize, 0, cudaImgHandle, 0);
    cuMemRelease(cudaImgHandle);

    d_output = (unsigned char*)d_ptr;
    CUmemAccessDesc accessDescriptor = {};
    accessDescriptor.location.id = cudaDevice;
    accessDescriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDescriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Apply the access descriptor to the whole VA range. Essentially enables Read-Write access to the range.
    cuMemSetAccess(d_ptr, imgSize, &accessDescriptor, 1);


}

 // need to fix
__global__ void resizeKernel(unsigned char *orig, unsigned char *resized, int w, int h, int w1, int h1) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= w || y >= h) {
        return;
    }

    unsigned char px1 = orig[((y*2 * w * 3) + x * 3 * 2) + 2];
    unsigned char py1 = orig[((y*2 * w * 3) + x * 3 * 2) + 1];
    unsigned char pz1 = orig[((y*2 * w * 3) + x * 3 * 2) + 0];

    // unsigned char px2 = orig[((y*2 + 1) * w  * 3) + (x * 3 * 2 + 1) + 2];
    // unsigned char py2 = orig[((y*2 + 1) * w  * 3) + (x * 3 * 2 + 1) + 1];
    // unsigned char pz2 = orig[((y*2 + 1) * w  * 3) + (x * 3 * 2 + 1) + 0];

    // resized[((y * w1 * 4) + x * 4) + 0] = (px1 + px2) / 2;
    // resized[((y * w1 * 4) + x * 4) + 1] = (py1 + py2) / 2;
    // resized[((y * w1 * 4) + x * 4) + 2] = (pz1 + pz2) / 2;
    // resized[((y * w1 * 4) + x * 4) + 3] = (unsigned char)0;

    resized[((y * w1 * 4) + x * 4) + 0] = px1;
    resized[((y * w1 * 4) + x * 4) + 1] = py1;
    resized[((y * w1 * 4) + x * 4) + 2] = pz1;
    resized[((y * w1 * 4) + x * 4) + 3] = (unsigned char)0;


}

int CudaFuns::cudaops(unsigned char *img, int w, int h, int w1, int h1) {

    allocateMem(w1 * h1 * 4);
    unsigned char *orig = NULL;
    cudaMalloc(&orig, sizeof(unsigned char) * w * h * 3);
    cudaMemcpy(orig, img, sizeof(unsigned char) * w * h * 3, cudaMemcpyHostToDevice);

    int count = 10;
    dim3 blocks((w1 + count)/ count, (h1 + count) / count);
    dim3 threads(count, count);
    resizeKernel<<<blocks, threads>>>(orig, d_output, w, h, w1, h1);
    cudaPeekAtLastError();
    cudaFree(orig);

    return imgShareableHandle;

}

