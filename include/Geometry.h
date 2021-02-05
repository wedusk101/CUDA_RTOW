#pragma once
#ifndef RTOW_GEOMETRY_H__
#define RTOW_GEOMETRY_H__

#include "device_launch_parameters.h"

#include "Vec3.h"

struct Material;
struct HitInfo;

struct Geometry
{
	__host__ __device__ virtual bool intersects(const Ray &ray, HitInfo &hitRec) const = 0;
	__host__ __device__ virtual Vec3 getNormal(const Vec3 &point) const = 0;

	Material *material;
	const char *name = "\nGeometry\n";	
};

struct HitInfo
{
	__host__ __device__ HitInfo() {}

	const Geometry *geometry;
	Vec3 hitPoint;
	Vec3 normal;
	float t;
};

struct Sphere : public Geometry
{
	__host__ __device__ Sphere(const Vec3 &c, const float &rad, Material *mat) : center(c), radius(rad)
	{
		material = mat;
	}

	__host__ __device__ Vec3 getNormal(const Vec3 &point) const override // returns the surface normal at a point
	{
		return (point - center) / radius;
	}

	__host__ __device__ bool intersects(const Ray &ray, HitInfo &hitRec) const override
	{
		const float eps = 1e-4;
		const Vec3 oc = ray.o - center;
		const float b = 2 * (ray.d % oc);
		const float a = ray.d % ray.d;
		const float c = (oc % oc) - (radius * radius);
		float delta = b * b - 4 * a * c;
		if (delta < eps) // discriminant is less than zero
			return false;
		delta = sqrt(delta);
		const float t0 = (-b + delta) / (2 * a);
		const float t1 = (-b - delta) / (2 * a);
		ray.t = (t0 < t1) ? t0 : t1;
		if (ray.t >= ray.tMin && ray.t <= ray.tMax)
		{
			ray.tMax = ray.t;
			hitRec.t = ray.tMax;
			hitRec.geometry = this;
			hitRec.hitPoint = ray.getPointAtParameter(ray.tMax);
			hitRec.normal = this->getNormal(hitRec.hitPoint);
			return true;
		}
		else
			return false;
	}


	Vec3 center;
	float radius;
};

struct Scene : public Geometry
{

	__host__ __device__ Scene(int lSize) : sceneSize(lSize), material(nullptr), currentSceneSize(0)
	{
		printf("\nInitializing scene...");
		cudaErrorCheck(cudaMallocManaged((void**)&geoList, sceneSize * sizeof(Geometry*)));
		printf("\nScene initialized successfully.\nMax scene size: %d, Current scene size: %d\n", sceneSize, currentSceneSize);
	}

	__host__ __device__ virtual bool intersects(const Ray &ray, HitInfo &hitRec) const override
	{
		bool hitStatus = false;

		// printf("\nIntersecting...\n");
		for (int i = 0; i < sceneSize; ++i)
		{
			printf(geoList[i]->name);
			if (geoList[i]->intersects(ray, hitRec))
				hitStatus = true;
		}

		return hitStatus;
	}

	__host__ __device__ bool addGeometry(Geometry *object)
	{
		if (currentSceneSize < sceneSize)
		{
			geoList[currentSceneSize++] = object;
			// printf(object->name);
			return true;
		}
		return false;
	}
	
	__host__ __device__ virtual Vec3 getNormal(const Vec3 &point) const override { return Vec3(); }

	Material *material;
	Geometry **geoList;
	int sceneSize;
	int currentSceneSize;
};

#endif // RTOW_GEOMETRY_H__