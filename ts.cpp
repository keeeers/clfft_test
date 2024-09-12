#include <stdlib.h>
#include <iostream>
#include <clFFT.h>
#include <cmath>
#include <stdio.h>
#include "timeKeeper.h"

using  namespace std;

TimeKeeper timedog;

const char* layoutToString(clfftLayout_ layout);

template< class T >
void testCLFFT(clfftLayout_ startLayout, clfftLayout_ intermediaryLayout);

unsigned int N=1;
int main(void)
{
    /*testCLFFT< float >(CLFFT_REAL, CLFFT_COMPLEX_INTERLEAVED);
    testCLFFT< double >(CLFFT_REAL, CLFFT_COMPLEX_INTERLEAVED);
    testCLFFT< float >(CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    testCLFFT< double >(CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);*/
    for (int i = 1; i <= 28; i++, N = N * 2) {
        cout << "running Test: i=" << i<< " N=" << N << endl;
        testCLFFT< float >(CLFFT_REAL, CLFFT_HERMITIAN_PLANAR);
        cout << "_____________________________" << endl;
    }
    clfftTeardown();
    //testCLFFT< double >(CLFFT_REAL, CLFFT_HERMITIAN_PLANAR);
    return 0;
}


const char* layoutToString(clfftLayout_ layout)
{
    switch (layout) {
        case CLFFT_COMPLEX_INTERLEAVED: return "Complex interleaved";
        case CLFFT_COMPLEX_PLANAR: return "Complex planar";
        case CLFFT_HERMITIAN_INTERLEAVED: return "Hermitian interleaved";
        case CLFFT_HERMITIAN_PLANAR: return "Hermitian planar";
        case CLFFT_REAL: return "Real";
        case ENDLAYOUT: return "Unknown";
    }
    return "Not found";
}


template< class T >
void testCLFFT(clfftLayout_ startLayout, clfftLayout_ intermediaryLayout)
{
    timedog.Init();
    bool success = true;
    cl_int err;
    cl_uint num_platforms = 1;
    cl_platform_id* platforms;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context _ctx = 0;
    cl_command_queue _queue = 0;
    cl_mem _bufferIn;
    cl_mem _bufferMid;
    T* X;

    clfftPlanHandle _planHandle;
    clfftDim clfftDim = CLFFT_1D;
    size_t clLengths[1] = { N };
    int choice_platform = 0;
    platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    err = clGetDeviceIDs(platforms[choice_platform], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    // char device_name[128];
    // err = clGetDeviceInfo(device, CL_DEVICE_NAME,
    //     sizeof(device_name), device_name,
    //     NULL);
    // printf("%s\n", device_name);
    props[1] = (cl_context_properties)platforms[choice_platform];
    _ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    cl_command_queue_properties properties = 0;
    _queue = clCreateCommandQueueWithProperties(_ctx, device, &properties, &err);

    //_queue = clCreateCommandQueue(_ctx, device, 0, &err);

    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    // Filling array
       //    unsigned int N1 = N/2 + 1;
    X = (T*)malloc(N * 2 * sizeof(T));
    for (unsigned int i = 0; i < 2 * N; ++i) X[i] = static_cast<T>(0);
    for (unsigned int i = 0; i < N; ++i) X[i] = static_cast<T>(i);
    timedog.dur_withPre("Env Data Init::");
    //Watcher
   /* for (unsigned int i = 0; i < 2 * N; ++i) { cout << X[i] << " "; }
    cout << endl;
   */ // Forward transform
    timedog.Init();
    err = clfftCreateDefaultPlan(&_planHandle, _ctx, clfftDim, clLengths);
    if (sizeof(T) == 4) err = clfftSetPlanPrecision(_planHandle, CLFFT_SINGLE);
    else err = clfftSetPlanPrecision(_planHandle, CLFFT_DOUBLE);
    err = clfftSetLayout(_planHandle, startLayout, intermediaryLayout);
    err = clfftSetResultLocation(_planHandle, CLFFT_OUTOFPLACE);
    timedog.dur_withPre("Ready to Bake:");
    err = clfftBakePlan(_planHandle, 1, &_queue, NULL, NULL);
    timedog.dur_withPre("Bake Plan:");
    _bufferIn = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
    _bufferMid = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
    timedog.dur_withPre("FFT Buffers Init:");
    cl_mem buffer_list[2];
    for (int i = 0; i < 2; i++) {
        buffer_list[i] = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
    }
    err = clEnqueueWriteBuffer(_queue, _bufferIn, CL_TRUE, 0, N * 2 * sizeof(T), X, 0, NULL, NULL);
    timedog.dur_withPre("FFT Init:");
    timedog.Init();
    err = clfftEnqueueTransform(_planHandle, CLFFT_FORWARD, 1, &_queue, 0, NULL, NULL, &_bufferIn, buffer_list, NULL);
    err = clFinish(_queue);
    timedog.dur_withPre("FFT Transform:");
    //err = clEnqueueReadBuffer(_queue, _bufferMid, CL_TRUE, 0, N * 2 * sizeof(T), X, 0, NULL, NULL);
    timedog.Init();
    T* Y1, * Y2;
    Y1 = (T*)malloc(N * 2 * sizeof(T));
    Y2 = (T*)malloc(N * 2 * sizeof(T));
    err = clEnqueueReadBuffer(_queue, buffer_list[0], CL_TRUE, 0, N * 2 * sizeof(T), Y1, 0, NULL, NULL);
    err = clEnqueueReadBuffer(_queue, buffer_list[1], CL_TRUE, 0, N * 2 * sizeof(T), Y2, 0, NULL, NULL);
    timedog.dur_withPre("FFT ReadBuffer:");
    /*for (int i = 0; i < N * 2; i++) {
        printf("(%f,%f) \n", Y1[i], Y2[i]);
    }*/


    // Backward transform
    timedog.Init();
    err = clfftCreateDefaultPlan(&_planHandle, _ctx, clfftDim, clLengths);
    if (sizeof(T) == 4) err = clfftSetPlanPrecision(_planHandle, CLFFT_SINGLE);
    else err = clfftSetPlanPrecision(_planHandle, CLFFT_DOUBLE);
    err = clfftSetLayout(_planHandle, intermediaryLayout, startLayout);
    err = clfftSetResultLocation(_planHandle, CLFFT_OUTOFPLACE);
    err = clfftBakePlan(_planHandle, 1, &_queue, NULL, NULL);
    _bufferIn = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
    _bufferMid = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
    /*  cl_mem buffer_list[2];
      for (int i = 0; i < 2; i++) {
          buffer_list[i] = clCreateBuffer(_ctx, CL_MEM_READ_WRITE, N * 2 * sizeof(T), NULL, &err);
      }*/
    err = clEnqueueWriteBuffer(_queue, _bufferMid, CL_TRUE, 0, N * 2 * sizeof(T), X, 0, NULL, NULL);
    timedog.dur_withPre("IFFT Init:");
    timedog.Init();
    err = clfftEnqueueTransform(_planHandle, CLFFT_BACKWARD, 1, &_queue, 0, NULL, NULL, buffer_list, &_bufferIn, NULL);
    err = clFinish(_queue);
    timedog.dur_withPre("IFFT Transform:");
    timedog.Init();
    err = clEnqueueReadBuffer(_queue, _bufferIn, CL_TRUE, 0, N * 2 * sizeof(T), X, 0, NULL, NULL);
    timedog.dur_withPre("IFFT ReadBuffer");
    for (unsigned int i = 0; i < 2 * N; ++i) {
        X[i] = round(X[i]);
    //    cout << X[i] << " ";
    }
    //cout << endl;
    // Checking we find the initial array values
    for (unsigned int i = 0; i < N; ++i) {
        if (X[i] != i) {
            success = false;
            break;
        }
    }
    timedog.Init();
    // Releasing memory
    /*clReleaseMemObject(_bufferIn);
    free(X);
    err = clfftDestroyPlan(&_planHandle);
    clfftTeardown();
    clReleaseCommandQueue(_queue);
    clReleaseContext(_ctx);
    timedog.dur_withPre("Env Data Release:");
    */// Displaying FFT success status
    std::cout << "CLFFT with type " << (sizeof(T) == 4 ? "float" : "double") << " and with layouts " <<
        layoutToString(startLayout) << " to " << layoutToString(intermediaryLayout) <<
        (success ? " succeeded " : " failed ") << std::endl;
}