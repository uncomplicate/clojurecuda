/*
 * Copyright 1993-2025 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#if !defined(__VECTOR_TYPES_H__)
#define __VECTOR_TYPES_H__

#if !defined(__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__)
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__
#endif

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#ifndef __DOXYGEN_ONLY__
#include "crt/host_defines.h"
#endif

/* NVRTC compiler defines these instead of in the header (to reduce compile time)
*/
#ifndef __CUDACC_RTC_BUILTIN_VECTOR_TYPES__

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/


#if defined(__NVCOMPILER)
# define __NV_SILENCE_HOST_DEPRECATION_BEGIN                                   \
  _Pragma("diag_suppress 1216")                                                \
  _Pragma("diag_suppress deprecated_entity_with_custom_message")
# define __NV_SILENCE_HOST_DEPRECATION_END                                     \
  _Pragma("diag_default 1216")                                                 \
  _Pragma("diag_default deprecated_entity_with_custom_message")
#elif defined(__GNUC__) || defined(__clang__)
# define __NV_SILENCE_HOST_DEPRECATION_BEGIN                                   \
  _Pragma("GCC diagnostic push")                                               \
  _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
# define __NV_SILENCE_HOST_DEPRECATION_END                                     \
  _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
# define __NV_SILENCE_HOST_DEPRECATION_BEGIN                                   \
  __pragma(warning(push))                                                      \
  __pragma(warning(disable:4996))
# define __NV_SILENCE_HOST_DEPRECATION_END                                     \
  __pragma(warning(pop))
#elif defined(__EDG__)
# define __NV_SILENCE_HOST_DEPRECATION_BEGIN                                   \
  _Pragma("warning push")                                                      \
  _Pragma("warning disable 1444")
# define __NV_SILENCE_HOST_DEPRECATION_END                                     \
  _Pragma("warning pop")
#else
  #define __NV_SILENCE_HOST_DEPRECATION_BEGIN
  #define __NV_SILENCE_HOST_DEPRECATION_END
#endif

#if defined(__NV_NO_VECTOR_DEPRECATION_DIAG)
# if defined(__CUDACC__)
#  pragma nv_diag_suppress vector_deprecation
# endif
# define __NV_SILENCE_DEPRECATION_BEGIN
# define __NV_SILENCE_DEPRECATION_END
#elif defined(__CUDACC__) && defined(_MSC_VER)
# define __NV_SILENCE_DEPRECATION_BEGIN                                       \
  __pragma(nv_diagnostic push)                                                \
  __pragma(nv_diag_suppress 20199)                                            \
  __pragma(nv_diag_suppress 1444)                                             \
  __NV_SILENCE_HOST_DEPRECATION_BEGIN
# define __NV_SILENCE_DEPRECATION_END                                         \
  __NV_SILENCE_HOST_DEPRECATION_END                                           \
  __pragma(nv_diagnostic pop)
#elif defined(__CUDACC__)
# define __NV_SILENCE_DEPRECATION_BEGIN                                       \
  _Pragma("nv_diagnostic push")                                               \
  _Pragma("nv_diag_suppress 20199")                                           \
  _Pragma("nv_diag_suppress 1444")                                            \
  __NV_SILENCE_HOST_DEPRECATION_BEGIN 
# define __NV_SILENCE_DEPRECATION_END                                         \
  __NV_SILENCE_HOST_DEPRECATION_END                                           \
  _Pragma("nv_diagnostic pop")
#else
#define __NV_SILENCE_DEPRECATION_BEGIN __NV_SILENCE_HOST_DEPRECATION_BEGIN
#define __NV_SILENCE_DEPRECATION_END __NV_SILENCE_HOST_DEPRECATION_END
#endif


#if defined(__NV_NO_VECTOR_DEPRECATION_DIAG)
# define __VECTOR_TYPE_DEPRECATED__(msg)
#elif defined(_WIN32)
# define __VECTOR_TYPE_DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __VECTOR_TYPE_DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __VECTOR_TYPE_DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if !defined(__CUDACC__) && !defined(__CUDACC_RTC__) && \
    defined(_WIN32) && !defined(_WIN64)

#pragma warning(push)
#pragma warning(disable: 4201 4408)

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ tag                      \
{                                                  \
    union                                          \
    {                                              \
        struct { members };                        \
        struct { long long int :1,:0; };           \
    };                                             \
}

#else /* !__CUDACC__ && !__CUDACC_RTC__ && _WIN32 && !_WIN64 */

#define __cuda_builtin_vector_align8(tag, members) \
struct __device_builtin__ __align__(8) tag         \
{                                                  \
    members                                        \
}

#endif /* !__CUDACC__ && !__CUDACC_RTC__ && _WIN32 && !_WIN64 */

struct __device_builtin__ char1
{
    signed char x;
};

struct __device_builtin__ uchar1
{
    unsigned char x;
};


struct __device_builtin__ __align__(2) char2
{
    signed char x, y;
};

struct __device_builtin__ __align__(2) uchar2
{
    unsigned char x, y;
};

struct __device_builtin__ char3
{
    signed char x, y, z;
};

struct __device_builtin__ uchar3
{
    unsigned char x, y, z;
};

struct __device_builtin__ __align__(4) char4
{
    signed char x, y, z, w;
};

struct __device_builtin__ __align__(4) uchar4
{
    unsigned char x, y, z, w;
};

struct __device_builtin__ short1
{
    short x;
};

struct __device_builtin__ ushort1
{
    unsigned short x;
};

struct __device_builtin__ __align__(4) short2
{
    short x, y;
};

struct __device_builtin__ __align__(4) ushort2
{
    unsigned short x, y;
};

struct __device_builtin__ short3
{
    short x, y, z;
};

struct __device_builtin__ ushort3
{
    unsigned short x, y, z;
};

__cuda_builtin_vector_align8(short4, short x; short y; short z; short w;);
__cuda_builtin_vector_align8(ushort4, unsigned short x; unsigned short y; unsigned short z; unsigned short w;);

struct __device_builtin__ int1
{
    int x;
};

struct __device_builtin__ uint1
{
    unsigned int x;
};

__cuda_builtin_vector_align8(int2, int x; int y;);
__cuda_builtin_vector_align8(uint2, unsigned int x; unsigned int y;);

struct __device_builtin__ int3
{
    int x, y, z;
};

struct __device_builtin__ uint3
{
    unsigned int x, y, z;
};

struct __device_builtin__ __builtin_align__(16) int4
{
    int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) uint4
{
    unsigned int x, y, z, w;
};

struct __device_builtin__ long1
{
    long int x;
};

struct __device_builtin__ ulong1
{
    unsigned long x;
};

#if defined(_WIN32)
__cuda_builtin_vector_align8(long2, long int x; long int y;);
__cuda_builtin_vector_align8(ulong2, unsigned long int x; unsigned long int y;);
#else /* !_WIN32 */

struct __device_builtin__ __align__(2*sizeof(long int)) long2
{
    long int x, y;
};

struct __device_builtin__ __align__(2*sizeof(unsigned long int)) ulong2
{
    unsigned long int x, y;
};

#endif /* _WIN32 */

struct __device_builtin__ long3
{
    long int x, y, z;
};

struct __device_builtin__ ulong3
{
    unsigned long int x, y, z;
};

struct __VECTOR_TYPE_DEPRECATED__("use long4_16a or long4_32a")
__device_builtin__ __builtin_align__(16) long4
{
    long int x, y, z, w;
};

struct __VECTOR_TYPE_DEPRECATED__("use ulong4_16a or ulong4_32a")
__device_builtin__ __builtin_align__(16) ulong4
{
    unsigned long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) long4_16a
{
    long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) ulong4_16a
{
    unsigned long int x, y, z, w;
};

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable: 4324)
#endif
struct __device_builtin__ __builtin_align__(32) long4_32a
{
    long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(32) ulong4_32a
{
    unsigned long int x, y, z, w;
};
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

struct __device_builtin__ float1
{
    float x;
};

#if !defined(__CUDACC__) && defined(__arm__) && \
    defined(__ARM_PCS_VFP) && __GNUC__ == 4 && __GNUC_MINOR__ == 6

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-pedantic"

struct __device_builtin__ __attribute__((aligned(8))) float2
{
    float x; float y; float __cuda_gnu_arm_ice_workaround[0];
};

#pragma GCC poison __cuda_gnu_arm_ice_workaround
#pragma GCC diagnostic pop

#else /* !__CUDACC__ && __arm__ && __ARM_PCS_VFP &&
         __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

__cuda_builtin_vector_align8(float2, float x; float y;);

#endif /* !__CUDACC__ && __arm__ && __ARM_PCS_VFP &&
          __GNUC__ == 4&& __GNUC_MINOR__ == 6 */

struct __device_builtin__ float3
{
    float x, y, z;
};

struct __device_builtin__ __builtin_align__(16) float4
{
    float x, y, z, w;
};

struct __device_builtin__ longlong1
{
    long long int x;
};

struct __device_builtin__ ulonglong1
{
    unsigned long long int x;
};

struct __device_builtin__ __builtin_align__(16) longlong2
{
    long long int x, y;
};

struct __device_builtin__ __builtin_align__(16) ulonglong2
{
    unsigned long long int x, y;
};

struct __device_builtin__ longlong3
{
    long long int x, y, z;
};

struct __device_builtin__ ulonglong3
{
    unsigned long long int x, y, z;
};

struct __VECTOR_TYPE_DEPRECATED__("use longlong4_16a or longlong4_32a")
__device_builtin__ __builtin_align__(16) longlong4
{
    long long int x, y, z ,w;
};

struct __device_builtin__ __builtin_align__(16) longlong4_16a
{
    long long int x, y, z ,w;
};

struct __device_builtin__ __builtin_align__(32) longlong4_32a
{
    long long int x, y, z ,w;
};

struct __VECTOR_TYPE_DEPRECATED__("use ulonglong4_16a or ulonglong4_32a")
__device_builtin__ __builtin_align__(16) ulonglong4
{
    unsigned long long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) ulonglong4_16a
{
    unsigned long long int x, y, z, w;
};

struct __device_builtin__ __builtin_align__(32) ulonglong4_32a
{
    unsigned long long int x, y, z, w;
};


struct __device_builtin__ double1
{
    double x;
};

struct __device_builtin__ __builtin_align__(16) double2
{
    double x, y;
};

struct __device_builtin__ double3
{
    double x, y, z;
};

struct __VECTOR_TYPE_DEPRECATED__("use double4_16a or double4_32a")
__device_builtin__ __builtin_align__(16) double4
{
    double x, y, z, w;
};

struct __device_builtin__ __builtin_align__(16) double4_16a
{
    double x, y, z, w;
};

struct __device_builtin__ __builtin_align__(32) double4_32a
{
    double x, y, z, w;
};

#if !defined(__CUDACC__) && defined(_WIN32) && !defined(_WIN64)

#pragma warning(pop)

#endif /* !__CUDACC__ && _WIN32 && !_WIN64 */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

typedef __device_builtin__ struct char1 char1;
typedef __device_builtin__ struct uchar1 uchar1;
typedef __device_builtin__ struct char2 char2;
typedef __device_builtin__ struct uchar2 uchar2;
typedef __device_builtin__ struct char3 char3;
typedef __device_builtin__ struct uchar3 uchar3;
typedef __device_builtin__ struct char4 char4;
typedef __device_builtin__ struct uchar4 uchar4;
typedef __device_builtin__ struct short1 short1;
typedef __device_builtin__ struct ushort1 ushort1;
typedef __device_builtin__ struct short2 short2;
typedef __device_builtin__ struct ushort2 ushort2;
typedef __device_builtin__ struct short3 short3;
typedef __device_builtin__ struct ushort3 ushort3;
typedef __device_builtin__ struct short4 short4;
typedef __device_builtin__ struct ushort4 ushort4;
typedef __device_builtin__ struct int1 int1;
typedef __device_builtin__ struct uint1 uint1;
typedef __device_builtin__ struct int2 int2;
typedef __device_builtin__ struct uint2 uint2;
typedef __device_builtin__ struct int3 int3;
typedef __device_builtin__ struct uint3 uint3;
typedef __device_builtin__ struct int4 int4;
typedef __device_builtin__ struct uint4 uint4;
typedef __device_builtin__ struct long1 long1;
typedef __device_builtin__ struct ulong1 ulong1;
typedef __device_builtin__ struct long2 long2;
typedef __device_builtin__ struct ulong2 ulong2;
typedef __device_builtin__ struct long3 long3;
typedef __device_builtin__ struct ulong3 ulong3;
__NV_SILENCE_DEPRECATION_BEGIN
typedef __device_builtin__ struct long4 __VECTOR_TYPE_DEPRECATED__("use long4_16a or long4_32a") long4;
typedef __device_builtin__ struct ulong4 __VECTOR_TYPE_DEPRECATED__("use ulong4_16a or ulong4_32a") ulong4;
__NV_SILENCE_DEPRECATION_END
typedef __device_builtin__ struct long4_16a long4_16a;
typedef __device_builtin__ struct ulong4_16a ulong4_16a;
typedef __device_builtin__ struct long4_32a long4_32a;
typedef __device_builtin__ struct ulong4_32a ulong4_32a;
typedef __device_builtin__ struct float1 float1;
typedef __device_builtin__ struct float2 float2;
typedef __device_builtin__ struct float3 float3;
typedef __device_builtin__ struct float4 float4;
typedef __device_builtin__ struct longlong1 longlong1;
typedef __device_builtin__ struct ulonglong1 ulonglong1;
typedef __device_builtin__ struct longlong2 longlong2;
typedef __device_builtin__ struct ulonglong2 ulonglong2;
typedef __device_builtin__ struct longlong3 longlong3;
typedef __device_builtin__ struct ulonglong3 ulonglong3;
__NV_SILENCE_DEPRECATION_BEGIN
typedef __device_builtin__ struct longlong4 __VECTOR_TYPE_DEPRECATED__("use longlong4_16a or longlong4_32a") longlong4;
typedef __device_builtin__ struct ulonglong4 __VECTOR_TYPE_DEPRECATED__("use ulonglong4_16a or ulonglong4_32a") ulonglong4;
__NV_SILENCE_DEPRECATION_END
typedef __device_builtin__ struct longlong4_16a longlong4_16a;
typedef __device_builtin__ struct ulonglong4_16a ulonglong4_16a;
typedef __device_builtin__ struct longlong4_32a longlong4_32a;
typedef __device_builtin__ struct ulonglong4_32a ulonglong4_32a;
typedef __device_builtin__ struct double1 double1;
typedef __device_builtin__ struct double2 double2;
typedef __device_builtin__ struct double3 double3;
__NV_SILENCE_DEPRECATION_BEGIN
typedef __device_builtin__ struct double4 __VECTOR_TYPE_DEPRECATED__("use double4_16a or double4_32a") double4;
__NV_SILENCE_DEPRECATION_END
typedef __device_builtin__ struct double4_16a double4_16a;
typedef __device_builtin__ struct double4_32a double4_32a;

#undef  __cuda_builtin_vector_align8

#endif /* !defined(__CUDACC_RTC_BUILTIN_VECTOR_TYPES__) */

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

struct __device_builtin__ dim3
{
    unsigned int x, y, z;
#if defined(__cplusplus)
#if __cplusplus >= 201103L || ( defined(_MSC_VER) && _MSC_VER >= 1900 )
    /* MSVC 2015 introduced support for constexpr constructors. A check in addition to the _cpluscplus macro comparison
       that uses the _MSC_VER macro is required because by default, Visual Studio always returns the value 199711L for 
       the __cplusplus preprocessor macro. */
    __host__ __device__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ constexpr operator uint3(void) const { return uint3{x, y, z}; }
#else
    __host__ __device__ dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
    __host__ __device__ dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
    __host__ __device__ operator uint3(void) const { uint3 t; t.x = x; t.y = y; t.z = z; return t; }
#endif
#endif /* __cplusplus */
};

typedef __device_builtin__ struct dim3 dim3;

#undef __VECTOR_TYPE_DEPRECATED__

#if defined(__UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__)
#undef __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#undef __UNDEF_CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS_VECTOR_TYPES_H__
#endif

#endif /* !__VECTOR_TYPES_H__ */
