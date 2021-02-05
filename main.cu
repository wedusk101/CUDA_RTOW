#include <cuda.h>
#include "cuda_runtime.h"

#include <fstream>
#include <cmath>
#include <iostream> 
#include <string>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>

#include "include/Utils.h"
#include "include/Vec3.h"
#include "include/Scene.h"
#include "include/Renderer.h"

int main(int argc, char* argv[])
{
	int width = 2560;
	int height = 1440;
	int spp = 1;
	int tx = 8;
	int ty = 8;
	int nThreads = 1024;

	// setup multithreading and benchmark parameters

	int nBenchLoops = 1;
	bool isBenchmark = false;

	for (int i = 0; i < argc; i++) // process command line args
	{
		if (!strcmp(argv[i], "-bench")) // usage -bench numberLoops
		{
			isBenchmark = true;
			if (i + 1 < argc)
			{
				if (isNumber(argv[i + 1]))
					nBenchLoops = atoi(argv[i + 1]); // number of times to loop in benchmark mode
				else
				{
					std::cout << "Invalid benchmark loop count provided. Using default value.\n";
					nBenchLoops = 5;
				}
			}
			else
			{
				std::cout << "Benchmark loop count not provided. Using default value.\n";
				nBenchLoops = 5;
			}
		}

		if (!strcmp(argv[i], "-width"))
		{
			if (i + 1 < argc)
			{
				if (isNumber(argv[i + 1]))
					width = atoi(argv[i + 1]);
				else
					std::cout << "Invalid image width provided. Using default value.\n";
			}
			else
				std::cout << "Image width not provided. Using default value.\n";
		}

		if (!strcmp(argv[i], "-height"))
		{
			if (i + 1 < argc)
			{
				if (isNumber(argv[i + 1]))
					height = atoi(argv[i + 1]);
				else
					std::cout << "Invalid image height provided. Using default value.\n";
			}
			else
				std::cout << "Image height not provided. Using default value.\n";
		}

		if (!strcmp(argv[i], "-spp"))
		{
			if (i + 1 < argc)
			{
				if (isNumber(argv[i + 1]))
					spp = atoi(argv[i + 1]);
				else
					std::cout << "Invalid sample count provided. Using default value.\n";
			}
			else
				std::cout << "Sample count not provided. Using default value.\n";
		}
	}

	if (argc == 1)
		std::cout << "Arguments not provided. Using default values.\n";

	// colors (R, G, B)
	const Vec3 white(1, 1, 1);
	const Vec3 black(0, 0, 0);
	const Vec3 red(1, 0, 0);
	const Vec3 green(0, 1, 0);
	const Vec3 blue(0, 0, 1);
	const Vec3 cyan(0, 1, 1);
	const Vec3 magenta(1, 0, 1);
	const Vec3 yellow(1, 1, 0);

	Camera *camera;
	Geometry **scene;
	Material **materials;
	int sceneSize = 3;
	// Scene *scene;
	// int matSize = 2;

	int numPixels = width * height;
	int fbSize = numPixels * sizeof(Vec3);

	Vec3 *fb;
	curandState *d_randState;

	cudaErrorCheck(cudaMallocManaged((void**)&d_randState, nThreads * sizeof(curandState)));
	cudaErrorCheck(cudaMallocManaged((void**)&fb, fbSize));
	cudaErrorCheck(cudaMallocManaged((void**)&camera, sizeof(Camera)));
	cudaErrorCheck(cudaMallocManaged((void**)&scene, sizeof(Geometry*)));
	cudaErrorCheck(cudaMallocManaged((void**)&materials, sizeof(Material*)));

	// cudaErrorCheck(cudaMallocManaged((void**)&scene, sizeof(Scene)));
	// scene = new (scene) Scene(sceneSize);

	dim3 threadsPerBlock(tx, ty);
	dim3 numBlocks(width / threadsPerBlock.x, height / threadsPerBlock.y);

	auto start = std::chrono::high_resolution_clock::now();

	if (isBenchmark)
		std::cout << "\nRunning in benchmark mode. Looping " << nBenchLoops << " times.\n";

	// std::cout << "\nRendering...\n";

	initRandState << <1, nThreads >> > (d_randState, time(NULL));
	initScene << <1, 1 >> > (width, height, camera, scene, sceneSize, materials);
	cudaErrorCheck(cudaDeviceSynchronize());
	cudaErrorCheck(cudaGetLastError());
	for (int run = 0; run < nBenchLoops; run++)
	{
		render << <numBlocks, threadsPerBlock >> > (fb, width, height, spp, camera, scene, sceneSize, d_randState);
		cudaErrorCheck(cudaGetLastError());
		cudaErrorCheck(cudaDeviceSynchronize());
	}

	auto stop = std::chrono::high_resolution_clock::now();

	std::ofstream out("result.ppm"); // creates a PPM image file for saving the rendered output
	out << "P3\n" << width << " " << height << "\n255\n";

	for (int i = 0; i < numPixels; ++i)
		out << (int)(255.99 * fb[i].x) << " " << (int)(255.99 * fb[i].y) << " " << (int)(255.99 * fb[i].z) << "\n"; // write out the pixel values

	std::cout << "\nTime taken was " << (std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)).count() << " milliseconds." << std::endl;
	cudaErrorCheck(cudaFree(fb));
	cudaErrorCheck(cudaFree(camera));
	cudaErrorCheck(cudaFree(scene));
	cudaErrorCheck(cudaFree(materials));
}