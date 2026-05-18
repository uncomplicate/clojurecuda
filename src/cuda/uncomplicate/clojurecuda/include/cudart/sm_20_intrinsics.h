/*
 * Copyright 1993-2021 NVIDIA Corporation.  All rights reserved.
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

//NOTE: For NVRTC, these declarations have been moved into the compiler (to reduce compile time)
#define EXCLUDE_FROM_RTC

#if !defined(__SM_20_INTRINSICS_H__)
#define __SM_20_INTRINSICS_H__

#if defined(__CUDACC_RTC__)
#define __SM_20_INTRINSICS_DECL__ __device__
#define __COMMON_INTRINSICS_DECL__ __device__
#else /* __CUDACC_RTC__ */
#define __SM_20_INTRINSICS_DECL__ static __inline__ __device__
#define __COMMON_INTRINSICS_DECL__ static __inline__ __host__ __device__
#endif /* __CUDACC_RTC__ */

#if defined(__cplusplus) && defined(__CUDACC__)

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/

#include "cuda_runtime_api.h"

#if !defined(__CUDA_ARCH__) && !defined(_NVHPC_CUDA)
#define __DEF_IF_HOST { }
#else  /* !__CUDA_ARCH__ && !_NVHPC_CUDA */
#define __DEF_IF_HOST ;
#endif /* __CUDA_ARCH__ || _NVHPC_CUDA */

#if defined(_WIN32)
# define __DEPRECATED__(msg) __declspec(deprecated(msg))
#elif (defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 5 && !defined(__clang__))))
# define __DEPRECATED__(msg) __attribute__((deprecated))
#else
# define __DEPRECATED__(msg) __attribute__((deprecated(msg)))
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on compute_70 and above, and should be replaced with "#x"_sync()."\
    "To continue using "#x"(), specify virtual architecture compute_60 when targeting sm_70 and above, for example, using the pair of compiler options: -arch=compute_60 -code=sm_70."
#elif defined(_NVHPC_CUDA)
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is not valid on cc70 and above, and should be replaced with "#x"_sync()."
#else
#define __WSB_DEPRECATION_MESSAGE(x) #x"() is deprecated in favor of "#x"_sync() and may be removed in a future release (Use -Wno-deprecated-declarations to suppress this warning)."
#endif

extern "C"
{
extern __device__ __device_builtin__ void                   __threadfence_system(void);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-to-nearest-even mode.
 *
 * Divides two floating-point values \p x by \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x / \p y.
 * - sign of the quotient \p x / \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __ddiv_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns NaN.
 * - __ddiv_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __ddiv_rn(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p x.
 * - __ddiv_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for finite \p y.
 * - __ddiv_rn(\p x, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __ddiv_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for \p y \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-towards-zero mode.
 *
 * Divides two floating-point values \p x by \p y in round-towards-zero mode.
 *
 * \return Returns \p x / \p y.
 * - sign of the quotient \p x / \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __ddiv_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns NaN.
 * - __ddiv_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __ddiv_rz(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p x.
 * - __ddiv_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for finite \p y.
 * - __ddiv_rz(\p x, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __ddiv_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for \p y \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-up mode.
 * 
 * Divides two floating-point values \p x by \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x / \p y.
 * - sign of the quotient \p x / \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __ddiv_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns NaN.
 * - __ddiv_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __ddiv_ru(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p x.
 * - __ddiv_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for finite \p y.
 * - __ddiv_ru(\p x, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __ddiv_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for \p y \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Divide two floating-point values in round-down mode.
 *
 * Divides two floating-point values \p x by \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x / \p y.
 * - sign of the quotient \p x / \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __ddiv_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns NaN.
 * - __ddiv_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __ddiv_rd(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p x.
 * - __ddiv_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for finite \p y.
 * - __ddiv_rd(\p x, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __ddiv_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for \p y \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __ddiv_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula
 *  in round-to-nearest-even mode.
 * 
 * Compute the reciprocal of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula
 *  in round-towards-zero mode.
 *
 * Compute the reciprocal of \p x in round-towards-zero mode.
 *
 * \return Returns 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula
 *  in round-up mode.
 * 
 * Compute the reciprocal of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula
 *  in round-down mode.
 * 
 * Compute the reciprocal of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
 * \cuda_math_formula \frac{1}{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __drcp_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula
 *  in round-to-nearest-even mode.
 * 
 * Compute the square root of \p x in round-to-nearest-even mode.
 *
 * \return Returns 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula
 *  in round-towards-zero mode.
 * 
 * Compute the square root of \p x in round-towards-zero mode.
 *
 * \return Returns 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula
 *  in round-up mode.
 * 
 * Compute the square root of \p x in round-up (to positive infinity) mode.
 *
 * \return Returns 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula
 *  in round-down mode.
 * 
 * Compute the square root of \p x in round-down (to negative infinity) mode.
 *
 * \return Returns 
 * \cuda_math_formula \sqrt{x} \end_cuda_math_formula.
 *
 * \note_accuracy_double_intrinsic
 * \note_requires_fermi
 */
extern __device__ __device_builtin__ double                __dsqrt_rd(double x);
extern __device__ __device_builtin__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__ballot)) unsigned int __ballot(int);
extern __device__ __device_builtin__ int                   __syncthreads_count(int);
extern __device__ __device_builtin__ int                   __syncthreads_and(int);
extern __device__ __device_builtin__ int                   __syncthreads_or(int);
extern __device__ __device_builtin__ long long int         clock64(void);


/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-to-nearest-even mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rn(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rn(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-down mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rd(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rd(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-up mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_ru(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_ru(float x, float y, float z);

/**
 * \ingroup CUDA_MATH_INTRINSIC_SINGLE
 * \brief Compute fused multiply-add operation in round-towards-zero mode, ignore \p -ftz=true compiler flag
 *
 * Behavior is the same as ::__fmaf_rz(\p x, \p y, \p z), the difference is in
 * handling denormalized inputs and outputs: \p -ftz compiler flag has no effect.
 */
extern __device__ __device_builtin__ float                  __fmaf_ieee_rz(float x, float y, float z);


// SM_13 intrinsics

/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a double as a 64-bit signed integer.
 *
 * Reinterpret the bits in the double-precision floating-point value \p x
 * as a signed 64-bit integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ long long int         __double_as_longlong(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret bits in a 64-bit signed integer as a double.
 *
 * Reinterpret the bits in the 64-bit signed integer value \p x as
 * a double-precision floating-point value.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ double                __longlong_as_double(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation in round-to-nearest-even mode.
 *
 * Computes the value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single ternary operation, rounding the
 * result once in round-to-nearest-even mode.
 *
 * \return Returns the rounded value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation.
 * - __fma_rn(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rn(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rn(\p x, \p y, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __fma_rn(\p x, \p y, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __fma_rn(\p x, \p y, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rn(\p x, \p y, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rn(\p x, \p y, \p z) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y + z \end_cuda_math_formula is exactly zero and \cuda_math_formula z \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 */
extern __device__ __device_builtin__ double                __fma_rn(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation in round-towards-zero mode.
 *
 * Computes the value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single ternary operation, rounding the
 * result once in round-towards-zero mode.
 *
 * \return Returns the rounded value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation.
 * - __fma_rz(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rz(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rz(\p x, \p y, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __fma_rz(\p x, \p y, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __fma_rz(\p x, \p y, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rz(\p x, \p y, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rz(\p x, \p y, \p z) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y + z \end_cuda_math_formula is exactly zero and \cuda_math_formula z \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 */
extern __device__ __device_builtin__ double                __fma_rz(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation in round-up mode.
 *
 * Computes the value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single ternary operation, rounding the
 * result once in round-up (to positive infinity) mode.
 *
 * \return Returns the rounded value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation.
 * - __fma_ru(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_ru(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_ru(\p x, \p y, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __fma_ru(\p x, \p y, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __fma_ru(\p x, \p y, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_ru(\p x, \p y, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_ru(\p x, \p y, \p z) returns \cuda_math_formula +0 \end_cuda_math_formula if \cuda_math_formula x \times y + z \end_cuda_math_formula is exactly zero and \cuda_math_formula z \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 */
extern __device__ __device_builtin__ double                __fma_ru(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Compute 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation in round-down mode.
 *
 * Computes the value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single ternary operation, rounding the
 * result once in round-down (to negative infinity) mode.
 *
 * \return Returns the rounded value of 
 * \cuda_math_formula x \times y + z \end_cuda_math_formula
 *  as a single operation.
 * - __fma_rd(
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rd(
 * \cuda_math_formula \pm 0 \end_cuda_math_formula
 * , 
 * \cuda_math_formula \pm \infty \end_cuda_math_formula
 * , \p z) returns NaN.
 * - __fma_rd(\p x, \p y, 
 * \cuda_math_formula -\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula +\infty \end_cuda_math_formula.
 * - __fma_rd(\p x, \p y, 
 * \cuda_math_formula +\infty \end_cuda_math_formula
 * ) returns NaN if
 * \cuda_math_formula x \times y \end_cuda_math_formula
 *  is an exact 
 * \cuda_math_formula -\infty \end_cuda_math_formula.
 * - __fma_rd(\p x, \p y, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rd(\p x, \p y, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula -0 \end_cuda_math_formula if \cuda_math_formula x \times y \end_cuda_math_formula is exact \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __fma_rd(\p x, \p y, \p z) returns \cuda_math_formula -0 \end_cuda_math_formula if \cuda_math_formula x \times y + z \end_cuda_math_formula is exactly zero and \cuda_math_formula z \neq 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 */
extern __device__ __device_builtin__ double                __fma_rd(double x, double y, double z);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-to-nearest-even mode.
 *
 * Adds two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x + \p y.
 * - __dadd_rn(\p x, \p y) is equivalent to __dadd_rn(\p y, \p x).
 * - __dadd_rn(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p x.
 * - __dadd_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dadd_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns NaN.
 * - __dadd_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dadd_rn(\p x, \p -x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-towards-zero mode.
 *
 * Adds two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x + \p y.
 * - __dadd_rz(\p x, \p y) is equivalent to __dadd_rz(\p y, \p x).
 * - __dadd_rz(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p x.
 * - __dadd_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dadd_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns NaN.
 * - __dadd_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dadd_rz(\p x, \p -x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-up mode.
 * 
 * Adds two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x + \p y.
 * - __dadd_ru(\p x, \p y) is equivalent to __dadd_ru(\p y, \p x).
 * - __dadd_ru(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p x.
 * - __dadd_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dadd_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns NaN.
 * - __dadd_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dadd_ru(\p x, \p -x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */ 
extern __device__ __device_builtin__ double                __dadd_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Add two floating-point values in round-down mode.
 *
 * Adds two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x + \p y.
 * - __dadd_rd(\p x, \p y) is equivalent to __dadd_rd(\p y, \p x).
 * - __dadd_rd(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p x.
 * - __dadd_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dadd_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns NaN.
 * - __dadd_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dadd_rd(\p x, \p -x) returns \cuda_math_formula -0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dadd_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-to-nearest-even mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x - \p y.
 * - __dsub_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p y.
 * - __dsub_rn(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \mp\infty \end_cuda_math_formula for finite \p x.
 * - __dsub_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dsub_rn(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dsub_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dsub_rn(\p x, \p x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-towards-zero mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x - \p y.
 * - __dsub_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p y.
 * - __dsub_rz(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \mp\infty \end_cuda_math_formula for finite \p x.
 * - __dsub_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dsub_rz(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dsub_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dsub_rz(\p x, \p x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-up mode.
 * 
 * Subtracts two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x - \p y.
 * - __dsub_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p y.
 * - __dsub_ru(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \mp\infty \end_cuda_math_formula for finite \p x.
 * - __dsub_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dsub_ru(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dsub_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dsub_ru(\p x, \p x) returns \cuda_math_formula +0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */ 
extern __device__ __device_builtin__ double                __dsub_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Subtract two floating-point values in round-down mode.
 *
 * Subtracts two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x - \p y.
 * - __dsub_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \p y) returns \cuda_math_formula \pm\infty \end_cuda_math_formula for finite \p y.
 * - __dsub_rd(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \mp\infty \end_cuda_math_formula for finite \p x.
 * - __dsub_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dsub_rd(\cuda_math_formula \pm\infty \end_cuda_math_formula, \cuda_math_formula \mp\infty \end_cuda_math_formula) returns \cuda_math_formula \pm\infty \end_cuda_math_formula.
 * - __dsub_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \mp 0 \end_cuda_math_formula) returns \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - __dsub_rd(\p x, \p x) returns \cuda_math_formula -0 \end_cuda_math_formula for finite \p x, including \cuda_math_formula \pm 0 \end_cuda_math_formula.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dsub_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-to-nearest-even mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-to-nearest-even mode.
 *
 * \return Returns \p x * \p y.
 * - sign of the product \p x * \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __dmul_rn(\p x, \p y) is equivalent to __dmul_rn(\p y, \p x).
 * - __dmul_rn(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __dmul_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dmul_rn(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p y.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rn(double x, double y);
/**      
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-towards-zero mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-towards-zero mode.
 *
 * \return Returns \p x * \p y.
 * - sign of the product \p x * \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __dmul_rz(\p x, \p y) is equivalent to __dmul_rz(\p y, \p x).
 * - __dmul_rz(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __dmul_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dmul_rz(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p y.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rz(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-up mode.
 * 
 * Multiplies two floating-point values \p x and \p y in round-up (to positive infinity) mode.
 *    
 * \return Returns \p x * \p y.
 * - sign of the product \p x * \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __dmul_ru(\p x, \p y) is equivalent to __dmul_ru(\p y, \p x).
 * - __dmul_ru(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __dmul_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dmul_ru(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p y.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_ru(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_DOUBLE
 * \brief Multiply two floating-point values in round-down mode.
 *
 * Multiplies two floating-point values \p x and \p y in round-down (to negative infinity) mode.
 *
 * \return Returns \p x * \p y.
 * - sign of the product \p x * \p y is XOR of the signs of \p x and \p y when neither inputs nor result are NaN.
 * - __dmul_rd(\p x, \p y) is equivalent to __dmul_rd(\p y, \p x).
 * - __dmul_rd(\p x, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns \cuda_math_formula \infty \end_cuda_math_formula of appropriate sign for \p x \cuda_math_formula \neq 0 \end_cuda_math_formula.
 * - __dmul_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \cuda_math_formula \pm\infty \end_cuda_math_formula) returns NaN.
 * - __dmul_rd(\cuda_math_formula \pm 0 \end_cuda_math_formula, \p y) returns \cuda_math_formula 0 \end_cuda_math_formula of appropriate sign for finite \p y.
 * - If either argument is NaN, NaN is returned.
 *
 * \note_accuracy_double_intrinsic
 * \note_nofma
 */
extern __device__ __device_builtin__ double                __dmul_rd(double x, double y);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-towards-zero mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rz(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a float in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a single-precision
 * floating-point value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ float                 __double2float_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ int                   __double2int_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ int                   __double2int_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ int                   __double2int_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned int          __double2uint_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ long long int          __double2ll_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ long long int          __double2ll_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to a signed 64-bit int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to a
 * signed 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ long long int          __double2ll_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-to-nearest-even mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-to-nearest-even mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_rn(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-up mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_ru(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a double to an unsigned 64-bit int in round-down mode.
 *
 * Convert the double-precision floating-point value \p x to an
 * unsigned 64-bit integer value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 * \note_fp_to_int_out_of_range_undefined
 */
extern __device__ __device_builtin__ unsigned long long int __double2ull_rd(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed int to a double.
 *
 * Convert the signed integer value \p x to a double-precision floating-point value.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __int2double_rn(int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned int to a double.
 *
 * Convert the unsigned integer value \p x to a double-precision floating-point value.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __uint2double_rn(unsigned int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rn(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rz(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-up mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_ru(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert a signed 64-bit int to a double in round-down mode.
 *
 * Convert the signed 64-bit integer value \p x to a double-precision floating-point
 * value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ll2double_rd(long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-to-nearest-even mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-to-nearest-even mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rn(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-towards-zero mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-towards-zero mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rz(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-up mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-up (to positive infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_ru(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Convert an unsigned 64-bit int to a double in round-down mode.
 *
 * Convert the unsigned 64-bit integer value \p x to a double-precision floating-point
 * value in round-down (to negative infinity) mode.
 * \return Returns converted value.
 */
extern __device__ __device_builtin__ double                 __ull2double_rd(unsigned long long int x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high 32 bits in a double as a signed integer.
 *
 * Reinterpret the high 32 bits in the double-precision floating-point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ int                    __double2hiint(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret low 32 bits in a double as a signed integer.
 *
 * Reinterpret the low 32 bits in the double-precision floating-point value \p x
 * as a signed integer.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ int                    __double2loint(double x);
/**
 * \ingroup CUDA_MATH_INTRINSIC_CAST
 * \brief Reinterpret high and low 32-bit integer values as a double.
 *
 * Reinterpret the integer value of \p hi as the high 32 bits of a 
 * double-precision floating-point value and the integer value of \p lo
 * as the low 32 bits of the same double-precision floating-point value.
 * \return Returns reinterpreted value.
 */
extern __device__ __device_builtin__ double                 __hiloint2double(int hi, int lo);


}

/*******************************************************************************
*                                                                              *
*                                                                              *
*                                                                              *
*******************************************************************************/
__SM_20_INTRINSICS_DECL__ __DEPRECATED__(__WSB_DEPRECATION_MESSAGE(__ballot)) unsigned int ballot(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ int syncthreads_count(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ bool syncthreads_and(bool pred) __DEF_IF_HOST

__SM_20_INTRINSICS_DECL__ bool syncthreads_or(bool pred) __DEF_IF_HOST

#undef __DEPRECATED__
#undef __WSB_DEPRECATION_MESSAGE

__SM_20_INTRINSICS_DECL__ unsigned int __isGlobal(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isShared(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isConstant(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ unsigned int __isLocal(const void *ptr) __DEF_IF_HOST
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
__SM_20_INTRINSICS_DECL__ unsigned int __isGridConstant(const void *ptr) __DEF_IF_HOST
#endif  /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700) */
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_global(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_shared(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_constant(const void *ptr) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_local(const void *ptr) __DEF_IF_HOST
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
__SM_20_INTRINSICS_DECL__ size_t __cvta_generic_to_grid_constant(const void *ptr) __DEF_IF_HOST
#endif  /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700) */

__SM_20_INTRINSICS_DECL__ void * __cvta_global_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_shared_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_constant_to_generic(size_t rawbits) __DEF_IF_HOST
__SM_20_INTRINSICS_DECL__ void * __cvta_local_to_generic(size_t rawbits) __DEF_IF_HOST
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700)
__SM_20_INTRINSICS_DECL__ void * __cvta_grid_constant_to_generic(size_t rawbits) __DEF_IF_HOST
#endif  /* !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 700) */

// notice: update documentation for __nv_bswap*() when more host compilers are supported
#if !defined(__CUDA_ARCH__) && !defined(_NVHPC_CUDA)
/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the order of bytes of the 16-bit unsigned integer.
 *
 * Reverse the order of bytes of \p x . Only supported in MSVC and other host 
 * compilers which define the `__GNUC__` macro, such as GCC and CLANG.
 *
 * \return Returns \p x with the order of bytes reversed.
 */
__COMMON_INTRINSICS_DECL__ unsigned short __nv_bswap16(unsigned short x) {
#if defined(__GNUC__)
    return __builtin_bswap16(x);
#elif defined(_WIN32)
    return _byteswap_ushort(x);
#else
#error "unsupported platform"
#endif /* defined(__GNUC__) */
}

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the order of bytes of the 32-bit unsigned integer.
 *
 * Reverse the order of bytes of \p x . Only supported in MSVC and other host 
 * compilers which define the `__GNUC__` macro, such as GCC and CLANG.
 *
 * \return Returns \p x with the order of bytes reversed.
 */
__COMMON_INTRINSICS_DECL__ unsigned int __nv_bswap32(unsigned int x) {
#if defined(__GNUC__)
    return __builtin_bswap32(x);
#elif defined(_WIN32)
    unsigned long ret = _byteswap_ulong(static_cast<unsigned long>(x));
    return static_cast<unsigned int>(ret);
#else
#error "unsupported platform"
#endif /* defined(__GNUC__) */
}

/**
 * \ingroup CUDA_MATH_INTRINSIC_INT
 * \brief Reverse the order of bytes of the 64-bit unsigned integer.
 *
 * Reverse the order of bytes of \p x . Only supported in MSVC and other host 
 * compilers which define the `__GNUC__` macro, such as GCC and CLANG.
 *
 * \return Returns \p x with the order of bytes reversed.
 */
__COMMON_INTRINSICS_DECL__ unsigned long long __nv_bswap64(unsigned long long x) {
#if defined(__GNUC__)
    return __builtin_bswap64(x);
#elif defined(_WIN32)
    unsigned __int64 ret = _byteswap_uint64(static_cast<unsigned __int64>(x));
    return static_cast<unsigned long long>(ret);
#else
#error "unsupported platform"
#endif /* defined(__GNUC__) */
}
#else
__COMMON_INTRINSICS_DECL__ unsigned short __nv_bswap16(unsigned short in);
__COMMON_INTRINSICS_DECL__ unsigned int __nv_bswap32(unsigned int in);
__COMMON_INTRINSICS_DECL__ unsigned long long __nv_bswap64(unsigned long long in);
#endif /* !defined(__CUDA_ARCH__) */

#endif /* __cplusplus && __CUDACC__ */

#undef __DEF_IF_HOST
#undef __SM_20_INTRINSICS_DECL__
#undef __COMMON_INTRINSICS_DECL__

#if (!defined(__CUDACC_RTC__) && defined(__CUDA_ARCH__)) || defined(_NVHPC_CUDA)
#include "sm_20_intrinsics.hpp"
#endif /* (!__CUDACC_RTC__ && __CUDA_ARCH__) || _NVHPC_CUDA */
#endif /* !__SM_20_INTRINSICS_H__ */

#undef EXCLUDE_FROM_RTC
