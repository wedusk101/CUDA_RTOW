#pragma once
#ifndef RTOW_UTILS_H__
#define RTOW_UTILS_H__

#include <iostream>
#include "cuda_runtime.h"
#include "curand_kernel.h"

#include "Vec3.h"

void errorCheck(cudaError_t code, const char *func, const char *fileName, const int line)
{
	if (code)
	{
		std::cerr << "CUDA error = " << (int)code << " in file: " <<
			fileName << " function: " << func << " on line: " << line << "\n";
		std::cerr << "Error details: " << cudaGetErrorString(code) << std::endl;
		cudaDeviceReset();
		std::exit(EXIT_FAILURE);
	}
}

#define cudaErrorCheck(arg) errorCheck( (arg), #arg, __FILE__, __LINE__ )

int isNumber(char *input)
{
	int len = strlen(input);
	for (int i = 0; i < len; ++i)
		if (!isdigit(input[i]))
			return 0;

	return 1;
}

__global__ void initRandState(curandState *randState, unsigned long seed)
{
	int idx = threadIdx.x;
	curand_init(seed, idx, 0, &randState[idx]);
}

__device__ Vec3 getRandomDirUnitSphere(curandState *globalState)
{
	int idx = threadIdx.x;
	curandState localState = globalState[idx];

	float z = 2 * curand_uniform(&localState) - 1;
	float t = 2 * 3.1415926 * curand_uniform(&localState);
	float r = sqrt(1 - z * z);
	float x = r * cos(t);
	float y = r * sin(t);
	globalState[idx] = localState;

	return Vec3(x, y, z).getNormalized();
}

#endif // RTOW_UTILS_H__
