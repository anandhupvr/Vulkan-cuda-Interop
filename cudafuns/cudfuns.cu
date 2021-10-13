#include "cudafuns.h"




#define ROUND_UP_TO_GRANULARITY(x, n) (((x + n - 1) / n) * n)

// CUmemAllocationHandleType ipcHandleTypeFlag = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;


// This will output the proper CUDA error strings in the event that a CUDA host
// call returns an error
#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line) {
  if (CUDA_SUCCESS != err) {
    const char *errorStr = NULL;
    cuGetErrorString(err, &errorStr);
    fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
    exit(EXIT_FAILURE);
  }
}
#endif


// Windows-specific LPSECURITYATTRIBUTES
void getDefaultSecurityDescriptorT(CUmemAllocationProp *prop) {
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

/**
 * @brief      CUDA safe call.
 *
 * @param[in]  err          The error
 * @param[in]  msg          The message
 * @param[in]  file_name    The file name
 * @param[in]  line_number  The line number
 */
static inline void _safe_cuda_call(cudaError err, const char* msg, const char* file_name, const int line_number) {
	if(err!=cudaSuccess) {
		fprintf(stderr,"%s\n\nFile: %s\n\nLine Number: %d\n\nReason: %s\n",msg,file_name,line_number,cudaGetErrorString(err));
		std::cin.get();
		exit(EXIT_FAILURE);
	}
}

/// Safe call macro.
#define SAFE_CALL(call,msg) _safe_cuda_call((call),(msg),__FILE__,__LINE__)


void CudaFuns::allocateMem(){

}

void CudaFuns::justCopy(const cv::Mat& input, unsigned char* output) {

}

