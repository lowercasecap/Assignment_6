#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Minimal OpenCL header definitions
#ifdef _WIN32
    #ifdef __MINGW32__
        #define CL_API_ENTRY
        #define CL_API_CALL
        #define CL_CALLBACK
    #else
        #define CL_API_ENTRY __stdcall
        #define CL_API_CALL __stdcall
        #define CL_CALLBACK __stdcall
    #endif
#else
    #define CL_API_ENTRY
    #define CL_API_CALL
    #define CL_CALLBACK
#endif

#include <stddef.h>
#include <stdint.h>

typedef int cl_int;
typedef unsigned int cl_uint;
typedef unsigned long long cl_ulong;
typedef struct _cl_platform_id * cl_platform_id;
typedef struct _cl_device_id * cl_device_id;
typedef struct _cl_context * cl_context;
typedef struct _cl_command_queue * cl_command_queue;
typedef struct _cl_mem * cl_mem;
typedef struct _cl_program * cl_program;
typedef struct _cl_kernel * cl_kernel;
typedef struct _cl_event * cl_event;
typedef unsigned int cl_bool;
typedef unsigned int cl_bitfield;
typedef cl_uint cl_device_type;
typedef cl_bitfield cl_mem_flags;
typedef cl_uint cl_program_build_info;
typedef cl_uint cl_profiling_info;

#define CL_SUCCESS 0
#define CL_DEVICE_TYPE_GPU (1 << 2)
#define CL_DEVICE_TYPE_CPU (1 << 1)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_WRITE_ONLY (1 << 1)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_PROGRAM_BUILD_LOG 0x1183
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROFILING_COMMAND_QUEUED 0x1280
#define CL_PROFILING_COMMAND_SUBMIT 0x1281
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_TRUE 1
#define CL_FALSE 0

extern CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint, cl_platform_id *, cl_uint *);
extern CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
extern CL_API_ENTRY cl_context CL_API_CALL clCreateContext(void *, cl_uint, const cl_device_id *, void *, void *, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context);
extern CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueue(cl_context, cl_device_id, cl_uint, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(cl_command_queue);
extern CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context, cl_mem_flags, size_t, void *, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem);
extern CL_API_ENTRY cl_program CL_API_CALL clCreateProgramWithSource(cl_context, cl_uint, const char **, const size_t *, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(cl_program, cl_uint, const cl_device_id *, const char *, void *, void *);
extern CL_API_ENTRY cl_int CL_API_CALL clGetProgramBuildInfo(cl_program, cl_device_id, cl_program_build_info, size_t, void *, size_t *);
extern CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program);
extern CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program, const char *, cl_int *);
extern CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel);
extern CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel, cl_uint, size_t, const void *);
extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueNDRangeKernel(cl_command_queue, cl_kernel, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint, const cl_event *, cl_event *);
extern CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue);
extern CL_API_ENTRY cl_int CL_API_CALL clEnqueueReadBuffer(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void *, cl_uint, const cl_event *, cl_event *);
extern CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(cl_event, cl_profiling_info, size_t, void *, size_t *);

// OpenCL kernel source code - modified for 1D work and timing
const char* kernelSource = 
"__kernel void matrixMultiply(__global const float* A, __global const float* B, __global float* C, __global ulong* timings, const int N) {\n"
"    int threadId = get_global_id(0);\n"
"    \n"
"    if (threadId < N * N) {\n"
"        int row = threadId / N;\n"
"        int col = threadId % N;\n"
"        \n"
"        float sum = 0.0f;\n"
"        for (int k = 0; k < N; k++) {\n"
"            sum += A[row * N + k] * B[k * N + col];\n"
"        }\n"
"        C[threadId] = sum;\n"
"        \n"
"        // Store thread ID in timing buffer\n"
"        if (timings != 0) {\n"
"            timings[threadId] = threadId;\n"
"        }\n"
"    }\n"
"}\n";

// Function to read kernel source (for file-based approach, but we'll use inline)
char* readKernelFromFile(const char* filename, size_t* size) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;
    
    fseek(file, 0, SEEK_END);
    *size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char* source = (char*)malloc(*size + 1);
    fread(source, 1, *size, file);
    source[*size] = '\0';
    fclose(file);
    return source;
}

int main() {
    const int TOTAL_THREADS = 1000;  // Exactly 1000 threads
    const int N = 32;  // Matrix size: N x N (32x32 = 1024, but we'll use 1000 threads)
    const size_t matrixSize = N * N * sizeof(float);
    const size_t timingSize = TOTAL_THREADS * sizeof(unsigned long long);
    
    printf("OpenCL Matrix Multiplication\n");
    printf("Total threads: %d\n", TOTAL_THREADS);
    printf("Matrix size: %dx%d\n\n", N, N);
    
    // Allocate host memory
    float* A = (float*)malloc(matrixSize);
    float* B = (float*)malloc(matrixSize);
    float* C = (float*)malloc(matrixSize);
    unsigned long long* threadTimings = (unsigned long long*)malloc(timingSize);
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
        C[i] = 0.0f;
    }
    
    // Initialize timing array
    for (int i = 0; i < TOTAL_THREADS; i++) {
        threadTimings[i] = 0;
    }
    
    // OpenCL setup
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem bufferA, bufferB, bufferC, bufferTimings;
    cl_int err;
    cl_event* events = (cl_event*)malloc(TOTAL_THREADS * sizeof(cl_event));
    
    // Get platform
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error getting platform: %d\n", err);
        return 1;
    }
    
    // Get device (try GPU first, then CPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf("GPU not available, using CPU...\n");
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error getting device: %d\n", err);
            return 1;
        }
    }
    printf("Device selected successfully\n");
    
    // Create context
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating context: %d\n", err);
        return 1;
    }
    
    // Create command queue with profiling enabled
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating command queue: %d\n", err);
        return 1;
    }
    
    // Create program from source
    size_t kernelSize = strlen(kernelSource);
    program = clCreateProgramWithSource(context, 1, &kernelSource, &kernelSize, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating program: %d\n", err);
        return 1;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error building program: %d\n", err);
        // Get build log
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
        char* log = (char*)malloc(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log, NULL);
        fprintf(stderr, "Build log:\n%s\n", log);
        free(log);
        return 1;
    }
    printf("Program built successfully\n");
    
    // Create kernel
    kernel = clCreateKernel(program, "matrixMultiply", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating kernel: %d\n", err);
        return 1;
    }
    
    // Create buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             matrixSize, A, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer A: %d\n", err);
        return 1;
    }
    
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                             matrixSize, B, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer B: %d\n", err);
        return 1;
    }
    
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrixSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating buffer C: %d\n", err);
        return 1;
    }
    
    bufferTimings = clCreateBuffer(context, CL_MEM_WRITE_ONLY, timingSize, NULL, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error creating timing buffer: %d\n", err);
        return 1;
    }
    printf("Buffers created successfully\n");
    
    // Set kernel arguments (will be set for each thread)
    int n = N;
    
    printf("\nExecuting %d threads individually to time each one...\n", TOTAL_THREADS);
    printf("This may take a moment...\n\n");
    
    // Start CPU timing
    clock_t cpuStart = clock();
    
    // Execute each thread individually to get per-thread timing
    cl_ulong* threadStartTimes = (cl_ulong*)malloc(TOTAL_THREADS * sizeof(cl_ulong));
    cl_ulong* threadEndTimes = (cl_ulong*)malloc(TOTAL_THREADS * sizeof(cl_ulong));
    
    // Set kernel arguments once (they're the same for all threads)
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferTimings);
    err |= clSetKernelArg(kernel, 4, sizeof(int), &n);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error setting kernel arguments: %d\n", err);
        return 1;
    }
    
    // Execute each thread individually to get per-thread timing
    for (int threadId = 0; threadId < TOTAL_THREADS; threadId++) {
        // Execute single thread at offset threadId
        size_t globalSize = 1;
        size_t localSize = 1;
        size_t offset = threadId;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &globalSize, &localSize, 
                                     0, NULL, &events[threadId]);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "Error executing thread %d: %d\n", threadId, err);
            return 1;
        }
    }
    
    // Wait for all threads to complete
    clFinish(queue);
    
    clock_t cpuEnd = clock();
    double cpuTime = ((double)(cpuEnd - cpuStart)) / CLOCKS_PER_SEC;
    
    // Get timing for each thread
    printf("Collecting timing data for each thread...\n");
    for (int i = 0; i < TOTAL_THREADS; i++) {
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), 
                               &threadStartTimes[i], NULL);
        clGetEventProfilingInfo(events[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), 
                               &threadEndTimes[i], NULL);
    }
    
    // Calculate statistics
    double totalGpuTime = 0.0;
    double minTime = 1e9;
    double maxTime = 0.0;
    for (int i = 0; i < TOTAL_THREADS; i++) {
        double threadTime = (threadEndTimes[i] - threadStartTimes[i]) / 1e9;  // Convert to seconds
        totalGpuTime += threadTime;
        if (threadTime < minTime) minTime = threadTime;
        if (threadTime > maxTime) maxTime = threadTime;
    }
    double avgTime = totalGpuTime / TOTAL_THREADS;
    
    // Read results
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, matrixSize, C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error reading buffer: %d\n", err);
        return 1;
    }
    
    // Verify result
    printf("\nVerification:\n");
    printf("C[0] = %f\n", C[0]);
    
    // Print timing results for each thread
    printf("\n=== Per-Thread Timing Results ===\n");
    printf("CPU Time (wall clock): %.6f seconds\n", cpuTime);
    printf("Total GPU Time (sum of all threads): %.6f seconds\n", totalGpuTime);
    printf("Average time per thread: %.6f seconds (%.3f microseconds)\n", avgTime, avgTime * 1e6);
    printf("Minimum thread time: %.6f seconds (%.3f microseconds)\n", minTime, minTime * 1e6);
    printf("Maximum thread time: %.6f seconds (%.3f microseconds)\n", maxTime, maxTime * 1e6);
    
    printf("\n=== Individual Thread Times (in microseconds) ===\n");
    printf("Thread ID | Execution Time (us)\n");
    printf("---------|-------------------\n");
    for (int i = 0; i < TOTAL_THREADS; i++) {
        double threadTime = (threadEndTimes[i] - threadStartTimes[i]) / 1e3;  // Convert to microseconds
        printf("Thread %3d | %10.3f\n", i, threadTime);
    }
    
    printf("\n=== Summary Statistics ===\n");
    printf("Total threads: %d\n", TOTAL_THREADS);
    printf("Average time per thread: %.3f microseconds\n", avgTime * 1e6);
    printf("Total execution time: %.6f seconds\n", cpuTime);
    printf("Throughput: %.2f threads/second\n", TOTAL_THREADS / cpuTime);
    
    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseMemObject(bufferTimings);
    for (int i = 0; i < TOTAL_THREADS; i++) {
        // Note: In real OpenCL, we should release events, but our minimal header doesn't have that
    }
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    free(A);
    free(B);
    free(C);
    free(threadTimings);
    free(events);
    free(threadStartTimes);
    free(threadEndTimes);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}

