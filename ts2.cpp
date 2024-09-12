#include <stdio.h>
#include <stdlib.h>
#include "clfft.h"

cl_int err;
cl_platform_id platform_return;

int clfft_R2HP_FFT1(size_t N = 16)
{
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufX;
    cl_mem bufY;
    float* X;
    float* Y;
    cl_event event = NULL;
    int ret = 0;
    char platform_name[128];
    char device_name[128];

    /* FFT library related declarations */
    clfftPlanHandle planHandle;
    clfftDim dim = CLFFT_1D;
    size_t clLengths[1] = { N };

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf("Error getting platform IDs: %d\n", err);
        return -1;
    }

    size_t ret_param_size = 0;
    platform = platform_return;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME,
        sizeof(platform_name), platform_name,
        &ret_param_size);
    printf("Platform found: %s\n", platform_name);

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    err = clGetDeviceInfo(device, CL_DEVICE_NAME,
        sizeof(device_name), device_name,
        &ret_param_size);
    printf("Device found on the above platform: %s\n", device_name);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);

    cl_command_queue_properties properties = 0; // Use default properties
    queue = clCreateCommandQueueWithProperties(ctx, device, &properties, &err);

    /* Setup clFFT. */
    clfftSetupData fftSetup;
    err = clfftInitSetupData(&fftSetup);
    err = clfftSetup(&fftSetup);

    if (err != CLFFT_SUCCESS) {
        printf("Error setting up clFFT: %d\n", err);
        return -1;
    }

    int Ysize = (N / 2 + 1) * 2 * sizeof(float);
    /* Allocate host & initialize data. */
    X = (float*)malloc(N * sizeof(*X));
    Y = (float*)malloc(Ysize);

    /* Print input array */
    printf("\nPerforming R2HP FFT on an one-dimensional array of size N = %lu\n", (unsigned long)N);
    int print_iter = 0;
    while (print_iter < N) {
        float x = (float)print_iter;
        X[print_iter] = x / 1.1;
        printf("(%f)\n", X[print_iter]);
        print_iter++;
    }

    /* Prepare OpenCL memory objects and place data inside them. */
    bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, N * sizeof(*X), NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, Ysize, NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0,
        N * sizeof(*X), X, 0, NULL, NULL);

    /* Create a default plan for a real-to-Hermitian-planar FFT. */
    err = clfftCreateDefaultPlan(&planHandle, ctx, dim, clLengths);

    /* Set plan parameters. */
    err = clfftSetPlanPrecision(planHandle, CLFFT_SINGLE);

    // 注意：这里应该使用 CLFFT_HERMITIAN_PLANAR 而不是 CLFFT_COMPLEX_PLANAR
    err = clfftSetLayout(planHandle, CLFFT_REAL, CLFFT_HERMITIAN_PLANAR);

    err = clfftSetResultLocation(planHandle, CLFFT_OUTOFPLACE);

    /* Bake the plan. */
    err = clfftBakePlan(planHandle, 1, &queue, NULL, NULL);
    if (err != CLFFT_SUCCESS) {
        printf("Error baking plan: %d\n", err);
        return -1;
    }

    /* Execute the plan. */
    err = clfftEnqueueTransform(planHandle, CLFFT_FORWARD, 1, &queue, 0, NULL, NULL, &bufX, &bufY, NULL);
    if (err != CLFFT_SUCCESS) {
        printf("Error executing forward transform: %d\n", err);
        return -1;
    }

    /* Wait for calculations to be finished. */
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("Error finishing command queue: %d\n", err);
        return -1;
    }

    /* Fetch results of calculations. */
    err = clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, Ysize, Y, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading buffer: %d\n", err);
        return -1;
    }

    /* Print output array */
    print_iter = 0;
    printf("\n\nR2HP FFT result: \n");
    while (print_iter < N / 2 + 1) {
        printf("(%f, %f) ;;", Y[print_iter], Y[N / 2 + 1 + print_iter]);
        print_iter++;
    }
    printf("\n");

    // 创建一个新的计划用于 C2R 变换
    clfftPlanHandle planHandleBackward;
    err = clfftCreateDefaultPlan(&planHandleBackward, ctx, dim, clLengths);

    err = clfftSetPlanPrecision(planHandleBackward, CLFFT_SINGLE);

    // 注意：这里应该使用 CLFFT_HERMITIAN_PLANAR 作为输入布局
    err = clfftSetLayout(planHandleBackward, CLFFT_HERMITIAN_PLANAR, CLFFT_REAL);

    err = clfftSetResultLocation(planHandleBackward, CLFFT_OUTOFPLACE);

    /* Bake the plan. */
    err = clfftBakePlan(planHandleBackward, 1, &queue, NULL, NULL);
    if (err != CLFFT_SUCCESS) {
        printf("Error baking backward plan: %d\n", err);
        return -1;
    }

    /* Execute the plan. */
    err = clfftEnqueueTransform(planHandleBackward, CLFFT_BACKWARD, 1, &queue, 0, NULL, NULL, &bufY, &bufX, NULL);
    if (err != CLFFT_SUCCESS) {
        printf("Error executing backward transform: %d\n", err);
        return -1;
    }

    /* Wait for calculations to be finished. */
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("Error finishing command queue: %d\n", err);
        return -1;
    }

    /* Fetch results of calculations. */
    err = clEnqueueReadBuffer(queue, bufX, CL_TRUE, 0, N * sizeof(*X), X, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error reading buffer: %d\n", err);
        return -1;
    }

    /* Print output array */
    print_iter = 0;
    printf("\n\nC2R FFT result: \n");
    while (print_iter < N) {
        printf("(%f)\n", X[print_iter]);
        print_iter++;
    }
    printf("\n");

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufY);

    free(X);
    free(Y);

    /* Release the plans. */
    err = clfftDestroyPlan(&planHandle);
    if (err != CLFFT_SUCCESS) {
        printf("Error destroying plan: %d\n", err);
        return -1;
    }

    err = clfftDestroyPlan(&planHandleBackward);
    if (err != CLFFT_SUCCESS) {
        printf("Error destroying backward plan: %d\n", err);
        return -1;
    }

    /* Release clFFT library. */
    clfftTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}

// ... 其他函数 ...
/*
* 
* // 函数原型声明
void print_device_info(cl_device_id device);
void print_platform_info(cl_platform_id platform);

// 主要函数
void list_all_opencl_devices(void) {
cl_int err;
cl_uint num_platforms;
cl_platform_id* platforms;
cl_device_id* devices;
cl_uint num_devices;

深色版本
// 获取平台的数量
err = clGetPlatformIDs(0, NULL, &num_platforms);
if (err != CL_SUCCESS) {
    printf("Error: Failed to get the number of platforms! Error code: %d\n", err);
    return;
}

// 分配内存存储平台ID
platforms = (cl_platform_id*)malloc(num_platforms * sizeof(cl_platform_id));
if (!platforms) {
    printf("Error: Failed to allocate memory for platform IDs.\n");
    return;
}

// 获取所有平台ID
err = clGetPlatformIDs(num_platforms, platforms, NULL);
if (err != CL_SUCCESS) {
    printf("Error: Failed to get platform IDs! Error code: %d\n", err);
    free(platforms);
    return;
}

// 遍历每个平台
for (cl_uint i = 0; i < num_platforms; ++i) {
    // 打印平台信息
    print_platform_info(platforms[i]);

    // 获取设备的数量
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err != CL_SUCCESS && err != CL_DEVICE_NOT_FOUND) {
        printf("Error: Failed to get the number of devices! Error code: %d\n", err);
        continue;
    }

    // 分配内存存储设备ID
    devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
    if (!devices) {
        printf("Error: Failed to allocate memory for device IDs.\n");
        continue;
    }

    // 获取所有设备ID
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to get device IDs! Error code: %d\n", err);
        free(devices);
        continue;
    }

    // 遍历每个设备
    for (cl_uint j = 0; j < num_devices; ++j) {
       // printf("  Device %u:\n", j + 1);
        print_device_info(devices[j]);
    }

    free(devices);
}

platform_return = platforms[1];
free(platforms);
}

// 打印设备信息
void print_device_info(cl_device_id device) {
return;
char name[128];
size_t name_size;
cl_device_type type;
cl_uint compute_units;

深色版本
clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), &name, &name_size);
clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, NULL);

printf("    Device Name: %s\n", name);
printf("    Device Type: ");
if (type & CL_DEVICE_TYPE_CPU) {
    printf("CPU");
}
if (type & CL_DEVICE_TYPE_GPU) {
    if (type & CL_DEVICE_TYPE_CPU) {
        printf(", ");
    }
    printf("GPU");
}
if (type & CL_DEVICE_TYPE_ACCELERATOR) {
    if (type & (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU)) {
        printf(", ");
    }
    printf("Accelerator");
}
printf("\n    Compute Units: %u\n", compute_units);
}

// 打印平台信息
void print_platform_info(cl_platform_id platform) {
return;
char platform_name[128];
size_t platform_name_size;

深色版本
clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platform_name), &platform_name, &platform_name_size);
printf("Platform: %s\n", platform_name);
}
int main()
{
    list_all_opencl_devices();
    clfft_R2HP_FFT1();
    return 0;
}*/