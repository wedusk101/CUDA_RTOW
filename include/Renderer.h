#pragma once
#ifndef RTOW_RENDERER_H__
#define RTOW_RENDERER_H__

#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "Vec3.h"
#include "Scene.h"

__device__ Vec3 colorModulate(const Vec3 &lightColor, const Vec3 &objectColor) // performs component wise multiplication for colors  
{
	return Vec3(lightColor.x * objectColor.x, lightColor.y * objectColor.y, lightColor.z * objectColor.z);
}

__device__ void clamp(Vec3 &col)
{
	col.x = (col.x > 1) ? 1 : (col.x < 0) ? 0 : col.x;
	col.y = (col.y > 1) ? 1 : (col.y < 0) ? 0 : col.y;
	col.z = (col.z > 1) ? 1 : (col.z < 0) ? 0 : col.z;
}

__device__ void gammaCorrect(Vec3 &col)
{
	col.x = sqrt(col.x);
	col.y = sqrt(col.y);
	col.z = sqrt(col.z);
}

__device__ Vec3 bgColor(const Ray &ray, const int imgHeight)
{
	Vec3 azure(0.5f, 0.625f, 1.0f);
	Vec3 white(1.0f, 1.0f, 1.0f);
	float lerpParameter = (ray.o.y) / (float)imgHeight;
	return white * (1 - lerpParameter) + azure * lerpParameter;
}

__device__ bool intersectionTest(const Ray &ray, Geometry **scene, int sceneSize, HitInfo &hitRec)
{
	bool hitStatus = false;
	for (int i = 0; i < sceneSize; ++i)
		if (scene[i]->intersects(ray, hitRec))
			hitStatus = true;

	return hitStatus;
}

__device__ Vec3 getPixelColor(const Ray &cameraRay, Geometry **scene, int sceneSize, int width, int height, curandState *globalRandState)
{
	Vec3 pixelColor;
	HitInfo hitRec;		
	
	Ray inRay = cameraRay;
	Vec3 albedo(1.0f, 1.0f, 1.0f);
	for (int i = 0; i < 8; ++i)
	{
		if (intersectionTest(inRay, scene, sceneSize, hitRec))
		{
			Ray outRay;
			Vec3 attenuation;

			if (hitRec.geometry->material->scatter(inRay, outRay, hitRec, attenuation, globalRandState))
			{
				if (hitRec.geometry->material->isLambertian)
					albedo *= attenuation * (outRay.d.absDot(hitRec.normal)) * 0.5;
				else
					albedo *= attenuation * 0.5;

				inRay = outRay;
				pixelColor += albedo;
			}
			else
				return Vec3();
		}
		else
		{
			albedo *= bgColor(inRay, height);
			pixelColor += albedo;
			break;
		}
	}

	return pixelColor;
}

__global__ void render(Vec3 *fb, int width, int height, int spp, const Camera *camera, Geometry **scene, int sceneSize, curandState *globalRandState)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if ((x >= width) || (y >= height))
		return;

	int idx = threadIdx.x;
	Vec3 pixelColor;
	int index = y * width + x;
	curandState localState;
	
	for (int i = 0; i < spp; i++)
	{
		localState = globalRandState[idx];
		float r = curand_uniform(&localState);		
		Ray cameraRay = camera->getRay(x, y, r);
		pixelColor += getPixelColor(cameraRay, scene, sceneSize, width, height, globalRandState);
	}
	pixelColor /= (float)spp;	
	clamp(pixelColor);
	fb[index] = pixelColor;
	globalRandState[idx] = localState;
}

#endif