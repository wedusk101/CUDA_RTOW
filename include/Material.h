#pragma once
#ifndef RTOW_MATERIAL_H__
#define RTOW_MATERIAL_H__

#include "curand.h"

#include "Vec3.h"
#include "Utils.h"

struct Material
{
	__device__ virtual bool scatter(const Ray &in, Ray &out, const HitInfo &info, Vec3 &scatterColor, curandState *globalState) const = 0;
	Vec3 albedo;

	bool isLambertian = false;
};

struct Lambertian : public Material
{
	__device__ Lambertian(const Vec3 &color)
	{
		albedo = color;
		isLambertian = true;
	}

	__device__ bool scatter(const Ray &inRay, Ray &outRay, const HitInfo &info, Vec3 &scatterColor, curandState *globalState) const override
	{
		Vec3 target = info.hitPoint + info.normal + getRandomDirUnitSphere(globalState);
		outRay = Ray(info.hitPoint, (target - info.hitPoint).getNormalized());
		scatterColor = info.geometry->material->albedo;
		return true;
	}
};

struct Metal : public Material
{
	__device__ Metal(const Vec3 &color, float r_): roughness(r_)
	{
		albedo = color;
	}

	__device__ bool scatter(const Ray &inRay, Ray &outRay, const HitInfo &info, Vec3 &scatterColor, curandState *globalState) const override
	{
		Vec3 normal = info.normal;
		Vec3 reflectedRay = getReflectedVec(inRay.d, normal);
		outRay = Ray(info.hitPoint, reflectedRay + getRandomDirUnitSphere(globalState) * roughness);
		scatterColor = info.geometry->material->albedo;
		return (outRay.d).dot(normal) > 0;
	}

	float roughness;
};

#endif // RTOW_MATERIAL_H__