#pragma once
#ifndef RTOW_SCENE_H__
#define RTOW_SCENE_H__

#include "device_launch_parameters.h"

#include <cstdio>

#include "Vec3.h"
#include "Geometry.h"
#include "Material.h"

struct Light
{
	__host__ __device__ Light(const Vec3 &position_, const float &radius_, const Vec3 &color_, const float &intensity_) : position(position_), radius(radius_), color(color_), intensity(intensity_) {}

	Vec3 position;
	float radius;
	Vec3 color;
	float intensity;
};

struct Camera
{
	__host__ __device__ Camera(const Vec3 &lookFrom_, const Vec3 &lookAt_, float aperture_, Vec3 wUp, int xRes, int yRes, float fd_, float hfov_) :
		lookFrom(lookFrom_), lookAt(lookAt_), direction(lookAt - lookFrom), worldUp(wUp), aperture(aperture_),
		xResolution(xRes), yResolution(yRes), focusDistance(fd_), hfov(hfov_)
	{
		aspectRatio = xResolution / (float)yResolution;
		float halfWidth = std::tan(hfov * 0.5) * focusDistance;
		float halfHeight = halfWidth / aspectRatio;
		lensRadius = aperture * 0.5;
		localZ = direction.getNormalized();
		localX = (direction.cross(worldUp)).getNormalized();
		localY = (localZ.cross(localX)).getNormalized();
		bottomLeft = lookFrom + localZ * focusDistance + localY * halfHeight - localX * halfWidth;
		xVecNDC = (localX * halfWidth * 2) / xResolution;
		yVecNDC = (localY * halfHeight * 2) / yResolution;
	}

	__host__ __device__ Ray getRay(int x, int y, float rand) const
	{
		float offsetX = (float)x + rand;
		float offsetY = (float)y + rand;
		Vec3 origin = lookFrom;
		Vec3 target = bottomLeft + xVecNDC * offsetX - yVecNDC * offsetY;
		return Ray(origin, (target - origin).getNormalized());
	}

	__host__ __device__ void update()
	{
		direction = lookAt - lookFrom;
		aspectRatio = xResolution / (float)yResolution;
		float halfWidth = std::tan(hfov * 0.5) * focusDistance;
		float halfHeight = halfWidth / aspectRatio;
		lensRadius = aperture * 0.5;
		localZ = direction.getNormalized();
		localX = (direction.cross(worldUp)).getNormalized();
		localY = (localZ.cross(localX)).getNormalized();
		bottomLeft = lookFrom + localZ * focusDistance + localY * halfHeight - localX * halfWidth;
		xVecNDC = (localX * halfWidth * 2) / xResolution;
		yVecNDC = (localY * halfHeight * 2) / yResolution;
	}

	int xResolution;
	int yResolution;	

	Vec3 lookFrom;
	Vec3 direction;
	Vec3 lookAt;
	Vec3 worldUp;
	Vec3 bottomLeft;

	Vec3 localX;
	Vec3 localY;
	Vec3 localZ;

	Vec3 xVecNDC;
	Vec3 yVecNDC;

	float focusDistance;
	float lensRadius;
	float hfov;
	float aperture;
	float aspectRatio;
};

__global__ void initScene(int width, int height, Camera *camera, Geometry **scene, int SceneSize, Material **materials)
{
	if (threadIdx.x == 0 && blockIdx.x == 0)
	{
		Vec3 lookFrom(0, -100, 0);
		Vec3 lookAt(0, 100, 800);
		Vec3 worldUp(0, 1, 0);
		float aperture = 20.0f;
		float fd = 1.0f;
		float fov = 15 / 57.6;

		camera->lookFrom = lookFrom;
		camera->lookAt = lookAt;
		camera->aperture = aperture;
		camera->worldUp = worldUp;
		camera->xResolution = width;
		camera->yResolution = height;
		camera->focusDistance = fd;
		camera->hfov = fov;
		camera->update();

		materials[0] = new Lambertian(Vec3(0.5, 0.5, 0.5)); // gray diffuse
		materials[1] = new Lambertian(Vec3(1.0f, 0.0f, 0.0f)); // red diffuse
		materials[2] = new Lambertian(Vec3(0.0f, 1.0f, 0.0f)); // green diffuse
		materials[3] = new Lambertian(Vec3(0.0f, 0.0f, 1.0f)); // blue diffuse
		materials[4] = new Lambertian(Vec3(1.0f, 1.0f, 0.0f)); // yellow diffuse
		materials[5] = new Metal(Vec3(1.0f, 1.0f, 1.0f), 0.25); // steel 

		/*

		bool status = true;

		status = (*scene)->addGeometry((Geometry*)new Sphere(Vec3(0, 100, 800), 20.0f, materials[1]));
		status = (*scene)->addGeometry((Geometry*)new Sphere(Vec3(0, 520, 800), 400.0f, materials[0]));
		

		if (!status)
			printf("\nAdding geometry failed...");

		*/

		scene[0] = new Sphere(Vec3(0, 100, 800), 20.0f, materials[5]);
		scene[1] = new Sphere(Vec3(60, 105, 800), 20.0f, materials[2]);
		scene[2] = new Sphere(Vec3(0, 520, 800), 400.0f, materials[0]);
	}
}

#endif
