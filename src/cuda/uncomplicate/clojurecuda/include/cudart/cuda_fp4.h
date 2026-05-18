/*
 * Copyright 2024 NVIDIA Corporation.  All rights reserved.
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

#ifndef __CUDA_FP4_H__
#define __CUDA_FP4_H__

/* Set up function decorations */
#if defined(__CUDACC__)
#define __CUDA_FP4_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP4__ __host__ __device__
#define __CUDA_HOSTDEVICE_FP4_DECL__ static __host__ __device__ __inline__
#else /* !defined(__CUDACC__) */
#if defined(__GNUC__)
#define __CUDA_HOSTDEVICE_FP4_DECL__ static __attribute__((unused))
#else
#define __CUDA_HOSTDEVICE_FP4_DECL__ static
#endif /* defined(__GNUC__) */
#define __CUDA_HOSTDEVICE_FP4__
#endif /* defined(__CUDACC_) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#define __CPP_VERSION_AT_LEAST_11_FP4
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#define __CPP_VERSION_AT_LEAST_11_FP4
#endif

/* bring in fp6 types infrastructure and dependencies */
#include "cuda_fp6.h"

/**
 * \defgroup CUDA_MATH_INTRINSIC_FP4 FP4 Intrinsics
 * This section describes fp4 intrinsic functions.
 * To use these functions, include the header file \p cuda_fp4.h in your
 * program.
 *
 * \note Most of the operations defined here benefit from native HW support
 * when compiled for specific GPU targets (e.g. devices of compute capability 10.0a),
 * other targets use emulation path.
 *
 * The following macros are available to help users selectively enable/disable
 * various definitions present in the header file:
 * - \p __CUDA_NO_FP4_CONVERSIONS__ - If defined, this macro will prevent any
 * use of the C++ type conversions (converting constructors and conversion
 * operators) defined in the header.
 * - \p __CUDA_NO_FP4_CONVERSION_OPERATORS__ - If defined, this macro will
 * prevent any use of the  C++ conversion operators from \p fp4 to other types.
 */

/**
 * \defgroup CUDA_MATH_FP4_MISC FP4 Conversion and Data Movement
 * \ingroup CUDA_MATH_INTRINSIC_FP4
 * To use these functions, include the header file \p cuda_fp4.h in your
 * program.
 */

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief 8-bit \p unsigned \p integer
 * type abstraction used for \p fp4 floating-point
 * numbers storage.
 */
typedef __nv_fp8_storage_t __nv_fp4_storage_t;

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief 8-bit \p unsigned \p integer
 * type abstraction used for storage of pairs of
 * \p fp4 floating-point numbers.
 */
typedef __nv_fp8_storage_t __nv_fp4x2_storage_t;

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief 16-bit \p unsigned \p integer
 * type abstraction used for storage of tetrads of
 * \p fp4 floating-point numbers.
 */
typedef __nv_fp8x2_storage_t __nv_fp4x4_storage_t;

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Enumerates the possible
 * interpretations of the 4-bit values when referring to them as
 * \p fp4 types.
 */
typedef enum __nv_fp4_interpretation_t {
    __NV_E2M1, /**< Stands for \p fp4 numbers of \p e2m1 kind. */
} __nv_fp4_interpretation_t;

/* Forward-declaration of C-style APIs */

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input \p double precision \p x to \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input \p x to \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t
__nv_cvt_double_to_fp4(const double x,
                       const __nv_fp4_interpretation_t fp4_interpretation,
                       const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input vector of two \p double precision numbers packed
 * in \p double2 \p x into a vector of two values of \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input vector \p x to a vector of two \p fp4 values of the
 * kind specified by \p fp4_interpretation parameter, using
 * rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4x2_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t
__nv_cvt_double2_to_fp4x2(const double2 x,
                          const __nv_fp4_interpretation_t fp4_interpretation,
                          const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input \p single precision \p x to \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input \p x to \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter, using
 * rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t
__nv_cvt_float_to_fp4(const float x,
                      const __nv_fp4_interpretation_t fp4_interpretation,
                      const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input vector of two \p single precision numbers packed
 * in \p float2 \p x into a vector of two values of \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input vector \p x to a vector of two \p fp4 values of the
 * kind specified by \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4x2_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t
__nv_cvt_float2_to_fp4x2(const float2 x,
                         const __nv_fp4_interpretation_t fp4_interpretation,
                         const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input \p half precision \p x to \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input \p x to \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t
__nv_cvt_halfraw_to_fp4(const __half_raw x,
                        const __nv_fp4_interpretation_t fp4_interpretation,
                        const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input vector of two \p half precision numbers packed
 * in \p __half2_raw \p x into a vector of two values of \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input vector \p x to a vector of two \p fp4 values of the
 * kind specified by \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4x2_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t __nv_cvt_halfraw2_to_fp4x2(
    const __half2_raw x,
    const __nv_fp4_interpretation_t fp4_interpretation,
    const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input \p nv_bfloat16 precision \p x to \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input \p x to \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t __nv_cvt_bfloat16raw_to_fp4(
    const __nv_bfloat16_raw x,
    const __nv_fp4_interpretation_t fp4_interpretation,
    const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input vector of two \p nv_bfloat16 precision numbers packed
 * in \p __nv_bfloat162_raw \p x into a vector of two values of \p fp4 type of the
 * requested kind using specified rounding mode and saturating
 * the out-of-range values.
 *
 * \details Converts input vector \p x to a vector of two \p fp4 values of the
 * kind specified by \p fp4_interpretation parameter,
 * using rounding mode specified by \p rounding parameter.
 * Large out-of-range values saturate to MAXNORM of the same sign.
 * \p NaN input values result in positive MAXNORM.
 *
 * \returns
 * - The \p __nv_fp4x2_storage_t value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t
__nv_cvt_bfloat16raw2_to_fp4x2(
    const __nv_bfloat162_raw x,
    const __nv_fp4_interpretation_t fp4_interpretation,
    const enum cudaRoundMode rounding);

/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input \p fp4 \p x of the specified kind
 * to \p half precision.
 *
 * \details Converts input \p x of \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter
 * to \p half precision.
 *
 * \returns
 * - The \p __half_raw value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __half_raw
__nv_cvt_fp4_to_halfraw(const __nv_fp4_storage_t x,
                        const __nv_fp4_interpretation_t fp4_interpretation);
/**
 * \ingroup CUDA_MATH_FP4_MISC
 * \brief Converts input vector of two \p fp4 values of the specified kind
 * to a vector of two \p half precision values packed in \p __half2_raw
 * structure.
 *
 * \details Converts input vector \p x of \p fp4 type of the kind specified by
 * \p fp4_interpretation parameter
 * to a vector of two \p half precision values and returns as \p __half2_raw
 * structure.
 *
 * \returns
 * - The \p __half2_raw value holds the result of conversion.
 */
__CUDA_HOSTDEVICE_FP4_DECL__ __half2_raw
__nv_cvt_fp4x2_to_halfraw2(const __nv_fp4x2_storage_t x,
                           const __nv_fp4_interpretation_t fp4_interpretation);

#if defined(__cplusplus)

#define __CUDA_FP4_TYPES_EXIST__

/* Forward-declaration of structures defined in "cuda_fp4.hpp" */
struct __nv_fp4_e2m1;
struct __nv_fp4x2_e2m1;
struct __nv_fp4x4_e2m1;

#endif /* defined(__cplusplus) */

#include "cuda_fp4.hpp"

#undef __CUDA_FP4_DECL__
#undef __CUDA_HOSTDEVICE_FP4__
#undef __CUDA_HOSTDEVICE_FP4_DECL__

#if defined(__CPP_VERSION_AT_LEAST_11_FP4)
#undef __CPP_VERSION_AT_LEAST_11_FP4
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP4) */

#endif /* end of include guard: __CUDA_FP4_H__ */
