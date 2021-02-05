#pragma once
#ifndef RTOW_VEC3_H__
#define RTOW_VEC3_H__

#include <cmath>

#include "device_launch_parameters.h"

struct Vec3
{
	float x;
	float y;
	float z;

	__host__ __device__ Vec3() : x(0), y(0), z(0) {}
	__host__ __device__ Vec3(const float &x_, const float &y_, const float &z_) : x(x_), y(y_), z(z_) {}

	__host__ __device__ float getMagnitude() const
	{
		return sqrt(x * x + y * y + z * z);
	}

	__host__ __device__ Vec3 getNormalized() const
	{
		float mag = getMagnitude();
		return Vec3(x / mag, y / mag, z / mag);
	}

	__host__ __device__ Vec3 operator+(const Vec3 &v) const // addition
	{
		return Vec3(x + v.x, y + v.y, z + v.z);
	}

	__host__ __device__ Vec3 operator-(const Vec3 &v) const // subtraction
	{
		return Vec3(x - v.x, y - v.y, z - v.z);
	}

	__host__ __device__ Vec3 operator*(const float &c) const // scalar multiplication
	{
		return Vec3(c * x, c * y, c * z);
	}

	__host__ __device__ Vec3 operator/(const float &c) const // scalar division
	{
		return Vec3(x / c, y / c, z / c);
	}

	__host__ __device__ Vec3& operator+=(const Vec3 &v) // addition
	{
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	__host__ __device__ Vec3& operator/=(const float &c) // scalar division
	{
		x /= c;
		y /= c;
		z /= c;
		return *this;
	}

	__host__ __device__ Vec3& operator*=(const Vec3 &v) // modulate
	{
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}

	__host__ __device__ float operator%(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}

	__host__ __device__ Vec3 operator&(const Vec3 &v) const // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	__host__ __device__ float dot(const Vec3 &v) const // dot product
	{
		return x * v.x + y * v.y + z * v.z;
	}

	__host__ __device__ float absDot(const Vec3 &v) const // dot product
	{
		return max(0.0f, x * v.x + y * v.y + z * v.z);
	}

	__host__ __device__ Vec3 cross(const Vec3 &v) const // cross product
	{
		return Vec3(y * v.z - v.y * z, z * v.x - x * v.z, x * v.y - y * v.x);
	}
};

struct Ray
{
	__host__ __device__ Ray() : o(Vec3()), d(Vec3()), t(INT_MAX), tMin(0.1), tMax(INT_MAX) {}

	__host__ __device__ Ray(const Vec3 &o_, const Vec3 &d_) : o(o_), d(d_), t(INT_MAX), tMin(0.1), tMax(INT_MAX) {}

	__host__ __device__ Vec3 getPointAtParameter(float t) const // returns the surface normal at a point
	{
		return o + d * t;
	}

	Vec3 o; // origin
	Vec3 d; // direction
	mutable float t;
	float tMin;
	mutable float tMax;
};

__device__ Vec3 getReflectedVec(const Vec3 &inVec, const Vec3 &normal)
{
	float dir = inVec.dot(normal);

	if (dir < 0)
		return (inVec - normal * inVec.dot(normal) * 2).getNormalized(); //  R = L - 2(N.L)N
	else
		return (normal * inVec.dot(normal) * 2 - inVec).getNormalized();
}

#endif // RTOW_VEC3H__
