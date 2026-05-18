/*
 * Copyright 2024-2025 NVIDIA Corporation.  All rights reserved.
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

#if !defined(__CUDA_FP6_HPP__)
#define __CUDA_FP6_HPP__

#if !defined(__CUDA_FP6_H__)
#error "Do not include this file directly. Instead, include cuda_fp6.h."
#endif

/* C++ header for ::std::memcpy (used for type punning in host-side
 * implementations). When compiling as a CUDA source file memcpy is provided
 * implicitly. !defined(__CUDACC__) implies !defined(__CUDACC_RTC__).
 */
#if defined(__cplusplus) && !defined(__CUDACC__)
#include <cstring>
#elif !defined(__cplusplus) && !defined(__CUDACC__)
#include <string.h>
#endif /* defined(__cplusplus) && !defined(__CUDACC__) */

/*
 * Bring in the standard assertions header to enforce the subset
 * of rounding modes supported by the APIs defined here.
 * NOTE: NVRTC defines its own assert
 */
#if !defined (__CUDACC_RTC__)
#include <assert.h>
#endif

/* Set up structure-alignment attribute */
#if !(defined __CUDA_ALIGN__)
#if defined(__CUDACC__)
#define __CUDA_ALIGN__(align) __align__(align)
#else
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas"
 * is available) */
#if __cplusplus >= 201103L
#define __CUDA_ALIGN__(n)                                                      \
    alignas(n) /* C++11 kindly gives us a keyword for this */
#else          /* !defined(__CPP_VERSION_AT_LEAST_11_FP6)*/
#if defined(__GNUC__)
#define __CUDA_ALIGN__(n) __attribute__((aligned(n)))
#elif defined(_MSC_VER)
#define __CUDA_ALIGN__(n) __declspec(align(n))
#else
#define __CUDA_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */
#endif /* defined(__CUDACC__) */
#endif /* !(defined __CUDA_ALIGN__) */

#if !(defined __CPP_VERSION_AT_LEAST_11_FP6)
/* need c++11 for explicit operators */
#define __CUDA_NO_FP6_CONVERSION_OPERATORS__
#endif

#if !(defined __DOXYGEN_ONLY__)

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6_storage_t
__nv_cvt_double_to_fp6(const double x,
                       const __nv_fp6_interpretation_t fp6_interpretation,
                       const enum cudaRoundMode rounding) {
    unsigned char res;
    unsigned long long int xbits;

#if defined(__CUDACC__) || (!defined __cplusplus)
    (void)memcpy(&xbits, &x, sizeof(x));
#else
    (void)::std::memcpy(&xbits, &x, sizeof(x));
#endif
    unsigned char FP6_MAXNORM;
    unsigned char FP6_MANTISSA_MASK;
    unsigned short int FP6_EXP_BIAS;
    unsigned long long int FP6_SIGNIFICAND_BITS;
    unsigned long long int FP6_MINDENORM_O2;
    unsigned long long int FP6_OVERFLOW_THRESHOLD;
    unsigned long long int FP6_MINNORM;
    const unsigned long long int DP_INF_BITS = 0x7FF0000000000000ULL;

    switch (fp6_interpretation)
    {
        case __NV_E2M3:
            FP6_EXP_BIAS = 1U;
            FP6_SIGNIFICAND_BITS = 4ULL;
            FP6_MANTISSA_MASK = 0x7U;
            FP6_MINDENORM_O2 = 0x3FB0000000000000ULL; // mindenorm/2 = 2^-4
            FP6_OVERFLOW_THRESHOLD =
                0x401E000000000000ULL; // maxnorm = 7.5
            FP6_MAXNORM = 0x1FU;
            FP6_MINNORM = 0x3FF0000000000000ULL; // minnorm = 2^0
            break;
        case __NV_E3M2:
        default:
            FP6_EXP_BIAS = 3U;
            FP6_SIGNIFICAND_BITS = 3ULL;
            FP6_MANTISSA_MASK = 0x3U;
            FP6_MINDENORM_O2 = 0x3FA0000000000000ULL; // mindenorm/2 = 2^-5
            FP6_OVERFLOW_THRESHOLD =
                0x403C000000000000ULL; // maxnorm = 28
            FP6_MAXNORM = 0x1FU;
            FP6_MINNORM = 0x3FD0000000000000ULL; // minnorm = 2^-2
            break;
    }

    // 1/2 LSB of the target format, positioned in double precision mantissa
    // helpful in midpoints detection during round-to-nearest-even step
    const unsigned long long int FP6_DP_HALF_ULP =
        (unsigned long long int)1ULL << (53ULL - FP6_SIGNIFICAND_BITS - 1ULL);
    // prepare sign bit in target format
    unsigned char sign = (unsigned char)((xbits >> 63ULL) << 5U);
    // prepare exponent field in target format
    unsigned char exp =
        (unsigned char)((((unsigned short int)(xbits >> 52ULL)) & 0x7FFU) -
                        1023U + FP6_EXP_BIAS);
    // round mantissa to target format width, rounding towards zero
    unsigned char mantissa =
        (unsigned char)(xbits >> (53ULL - FP6_SIGNIFICAND_BITS)) &
        FP6_MANTISSA_MASK;
    unsigned long long int absx = xbits & 0x7FFFFFFFFFFFFFFFULL;

    if (absx <= FP6_MINDENORM_O2) {
        // zero or underflow
        res = 0U;
    } else if (absx > FP6_OVERFLOW_THRESHOLD) {
        // overflow or NaN
        if (absx > DP_INF_BITS)
        {
            // NaN converts to positive FP6_MAXNORM
            sign = 0U;
        }
        res = FP6_MAXNORM;
    } else if (absx >= FP6_MINNORM) {
        res = (unsigned char)((exp << (FP6_SIGNIFICAND_BITS - 1U)) | mantissa);
        // rounded-off bits
        unsigned long long int round =
            xbits & ((FP6_DP_HALF_ULP << 1ULL) - 1ULL);
        if (rounding == cudaRoundNearest)
        {
            // round-to-nearest-even adjustment
            if ((round > FP6_DP_HALF_ULP) ||
                ((round == FP6_DP_HALF_ULP) && (mantissa & 1U))) {
                res = (unsigned char)(res + 1U);
            }
        } else {
            assert(rounding == cudaRoundZero);
        }
    } else // Denormal range
    {
        unsigned char shift = (unsigned char)(1U - exp);
        // add implicit leading bit
        mantissa |= (unsigned char)(1U << (FP6_SIGNIFICAND_BITS - 1U));
        // additional round-off due to denormalization
        res = (unsigned char)(mantissa >> shift);

        if (rounding == cudaRoundNearest)
        {
            // rounded-off bits, including implicit leading bit
            unsigned long long int round =
                (xbits | ((unsigned long long int)1ULL << (53ULL - 1ULL))) &
                ((FP6_DP_HALF_ULP << (shift + 1ULL)) - 1ULL);
            // round-to-nearest-even adjustment
            if ((round > (FP6_DP_HALF_ULP << shift)) ||
                ((round == (FP6_DP_HALF_ULP << shift)) && (res & 1U))) {
                res = (unsigned char)(res + 1U);
            }
        } else {
            assert(rounding == cudaRoundZero);
        }
    }

    res |= sign;

    return (__nv_fp6_storage_t)res;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6x2_storage_t
__nv_cvt_double2_to_fp6x2(const double2 x,
                          const __nv_fp6_interpretation_t fp6_interpretation,
                          const enum cudaRoundMode rounding) {
    __nv_fp6x2_storage_t storage = (__nv_fp6x2_storage_t)__nv_cvt_double_to_fp6(
        x.y, fp6_interpretation, rounding);
    storage = (__nv_fp6x2_storage_t)(storage << 8U);
    storage = (__nv_fp6x2_storage_t)(storage |
                                     __nv_cvt_double_to_fp6(
                                         x.x, fp6_interpretation, rounding));
    return storage;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6_storage_t
__nv_cvt_float_to_fp6(const float x,
                      const __nv_fp6_interpretation_t fp6_interpretation,
                      const enum cudaRoundMode rounding) {
    __nv_fp6_storage_t res = 0U;
    assert((rounding == cudaRoundNearest) || (rounding == cudaRoundZero));
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest)
    {
        __nv_fp6x2_storage_t storage;
        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x), "f"(0.0f));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x), "f"(0.0f));
        }
        res = (__nv_fp6_storage_t)storage;
    } else
#endif
    {
        res = __nv_cvt_double_to_fp6((double)x, fp6_interpretation, rounding);
    }
    return res;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6x2_storage_t
__nv_cvt_float2_to_fp6x2(const float2 x,
                         const __nv_fp6_interpretation_t fp6_interpretation,
                         const enum cudaRoundMode rounding) {
    assert((rounding == cudaRoundNearest) || (rounding == cudaRoundZero));
    __nv_fp6x2_storage_t storage;
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest) {
        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x.x), "f"(x.y));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;}\n"
                : "=h"(storage)
                : "f"(x.x), "f"(x.y));
        }
    } else
#endif
    {
        storage = (__nv_fp6x2_storage_t)__nv_cvt_float_to_fp6(
            x.y, fp6_interpretation, rounding);
        storage = (__nv_fp6x2_storage_t)(storage << 8U);
        storage = (__nv_fp6x2_storage_t)(storage | __nv_cvt_float_to_fp6(
                                                       x.x,
                                                       fp6_interpretation, rounding));
    }
    return storage;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6_storage_t
__nv_cvt_halfraw_to_fp6(const __half_raw x,
                        const __nv_fp6_interpretation_t fp6_interpretation,
                        const enum cudaRoundMode rounding) {
    assert((rounding == cudaRoundNearest) || (rounding == cudaRoundZero));

    __nv_fp6_storage_t res = 0U;
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest) {
        unsigned int half2_storage = (unsigned int)(x.x);
        __nv_fp6x2_storage_t tmp;
        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        }
        res = (__nv_fp6_storage_t)tmp;
    } else
#endif
    {
        float fx = __internal_halfraw_to_float(x);
        res = __nv_cvt_float_to_fp6(fx, fp6_interpretation, rounding);
    }

    return res;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6x2_storage_t __nv_cvt_halfraw2_to_fp6x2(
    const __half2_raw x,
    const __nv_fp6_interpretation_t fp6_interpretation,
    const enum cudaRoundMode rounding) {
    assert((rounding == cudaRoundNearest) || (rounding == cudaRoundZero));
    __nv_fp6x2_storage_t tmp;

#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest) {
        unsigned int half2_storage;
        (void)memcpy(&half2_storage, &x, sizeof(x));

        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.f16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(half2_storage));
        }
    } else
#endif
    {
        __half_raw raw;
        raw.x = x.x;
        __nv_fp6_storage_t lo =
            __nv_cvt_halfraw_to_fp6(raw, fp6_interpretation, rounding);
        raw.x = x.y;
        __nv_fp6_storage_t hi =
            __nv_cvt_halfraw_to_fp6(raw, fp6_interpretation, rounding);
        tmp = hi;
        tmp = (__nv_fp6x2_storage_t)(tmp << 8U);
        tmp = (__nv_fp6x2_storage_t)(tmp | lo);
    }
    return tmp;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6_storage_t __nv_cvt_bfloat16raw_to_fp6(
    const __nv_bfloat16_raw x,
    const __nv_fp6_interpretation_t fp6_interpretation,
    const enum cudaRoundMode rounding) {

    __nv_fp6_storage_t res = 0U;
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest) {
        unsigned int bf162_storage = (unsigned int)(x.x);
        __nv_fp6x2_storage_t tmp;
        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.bf16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(bf162_storage));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.bf16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(bf162_storage));
        }
        res = (__nv_fp6_storage_t)tmp;
    } else
#endif
    {
        float fx = __internal_bf16raw_to_float(x);
        res = __nv_cvt_float_to_fp6(fx, fp6_interpretation, rounding);
    }

    return res;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __nv_fp6x2_storage_t
__nv_cvt_bfloat16raw2_to_fp6x2(
    const __nv_bfloat162_raw x,
    const __nv_fp6_interpretation_t fp6_interpretation,
    const enum cudaRoundMode rounding) {

    __nv_fp6x2_storage_t tmp;

#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    if (rounding == cudaRoundNearest) {
        unsigned int bf162_storage;
        (void)memcpy(&bf162_storage, &x, sizeof(x));

        if (fp6_interpretation == __NV_E3M2) {
            asm("{cvt.rn.satfinite.e3m2x2.bf16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(bf162_storage));
        } else {
            asm("{cvt.rn.satfinite.e2m3x2.bf16x2 %0, %1;}\n"
                : "=h"(tmp)
                : "r"(bf162_storage));
        }
    } else
#endif
    {
        __nv_bfloat16_raw raw;
        raw.x = x.y;
        tmp =
            (__nv_fp6x2_storage_t)__nv_cvt_bfloat16raw_to_fp6(raw,
                                            fp6_interpretation, rounding);
        tmp = (__nv_fp6x2_storage_t)(tmp << 8U);
        raw.x = x.x;
        tmp = (__nv_fp6x2_storage_t)(tmp |
                                        __nv_cvt_bfloat16raw_to_fp6(raw,
                                            fp6_interpretation, rounding));
    }
    return tmp;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __half2_raw
__nv_cvt_fp6x2_to_halfraw2(const __nv_fp6x2_storage_t x,
                           const __nv_fp6_interpretation_t fp6_interpretation);
__CUDA_HOSTDEVICE_FP6_DECL__ __half_raw
__nv_cvt_fp6_to_halfraw(const __nv_fp6_storage_t x,
                        const __nv_fp6_interpretation_t fp6_interpretation) {
    __half_raw res;
    res.x = 0U;
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    res.x =
        __nv_cvt_fp6x2_to_halfraw2((__nv_fp6x2_storage_t)x, fp6_interpretation)
            .x;
#else
    {
        unsigned short int ur = (unsigned short int)x;
        ur = (unsigned short int)(ur << 10U);

        unsigned short int sign = ur & 0x8000U;
        unsigned short int exponent;
        unsigned short int bias_difference;
        unsigned short int mantissa;

        if (fp6_interpretation == __NV_E3M2) {
            bias_difference = (unsigned short int)(15 - 3) << 10U;
            exponent = (unsigned short int)(((ur & 0x7000U) >> 2U) + bias_difference);
            mantissa = (ur & 0x0C00U) >> 2U;
        } else {
            //__NV_E2M3
            bias_difference = (unsigned short int)(15 - 1) << 10U;
            exponent = (unsigned short int)(((ur & 0x6000U) >> 3U) + bias_difference);
            mantissa = (ur & 0x1C00U) >> 3U;
        }

        if (exponent == bias_difference) {
            // zero or denormal
            if (mantissa != 0U) {
                // normalize
                mantissa = (unsigned short int)(mantissa << 1U);
                while ((mantissa & 0x0400U) == 0U) {
                    mantissa = (unsigned short int)(mantissa << 1U);
                    exponent = (unsigned short int)(exponent - 0x0400U);
                }
                // discard implicit leading bit
                mantissa &= 0x03FFU;
            } else { // Zero
                exponent = 0U;
            }
        }

        res.x = (sign | exponent) | mantissa;
    }
#endif
    return res;
}

__CUDA_HOSTDEVICE_FP6_DECL__ __half2_raw
__nv_cvt_fp6x2_to_halfraw2(const __nv_fp6x2_storage_t x,
                           const __nv_fp6_interpretation_t fp6_interpretation) {
    __half2_raw res;
#if (__CUDA_FP8_INTERNAL_CAN_RELY_ON_PTX_FOR_SHORTTYPESCVT__)
    unsigned int half2_storage;
    if (fp6_interpretation == __NV_E3M2) {
        asm("{cvt.rn.f16x2.e3m2x2 %0, %1;}\n" : "=r"(half2_storage) : "h"(x));
    } else {
        asm("{cvt.rn.f16x2.e2m3x2 %0, %1;}\n" : "=r"(half2_storage) : "h"(x));
    }
    (void)memcpy(&res, &half2_storage, sizeof(half2_storage));
#else
    res.x =
        __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)x, fp6_interpretation).x;
    res.y = __nv_cvt_fp6_to_halfraw((__nv_fp6_storage_t)(x >> 8U),
                                    fp6_interpretation)
                .x;
#endif
    return res;
}

#endif /* !(defined __DOXYGEN_ONLY__) */

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

/**
 * \defgroup CUDA_MATH_FP6_E3M2_STRUCT C++ struct for handling fp6 data type of e3m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6_E3M2_STRUCT
 * \brief __nv_fp6_e3m2 datatype
 *
 * \details This structure implements the datatype for handling
 * \p fp6 floating-point numbers of \p e3m2 kind:
 * with 1 sign, 3 exponent, 1 implicit and 2 explicit mantissa bits.
 * This encoding does not support Inf/NaN.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(1) __nv_fp6_e3m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6_E3M2_STRUCT
     * Storage variable contains the \p fp6 floating-point data.
     */
    __nv_fp6_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6_e3m2() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider FP types */
    /* Note we do avoid constructor init-list because of special host/device
     * compilation rules */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __half data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const __half f) {
        __x = __nv_cvt_halfraw_to_fp6(static_cast<__half_raw>(f),
                                      __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __nv_bfloat16 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const __nv_bfloat16 f) {
        __x = __nv_cvt_bfloat16raw_to_fp6(static_cast<__nv_bfloat16_raw>(f),
                                          __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const float f) {
        __x = __nv_cvt_float_to_fp6(f, __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const double f) {
        __x = __nv_cvt_double_to_fp6(f, __NV_E3M2, cudaRoundNearest);
    }

    /* Converts from integral */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p short \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__
    __nv_fp6_e3m2(const unsigned short int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const unsigned int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const unsigned long int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p long \p long \p int data type, relies on
     * \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__
    __nv_fp6_e3m2(const unsigned long long int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p short \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const short int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p long \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const long int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p long \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e3m2(const long long int val) {
        __x = static_cast<__nv_fp6_e3m2>(static_cast<float>(val)).__x;
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening FP converts */
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __half data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp6_to_halfraw(__x, __NV_E3M2));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float() const {
        return __internal_halfraw_to_float(
            __nv_cvt_fp6_to_halfraw(__x, __NV_E3M2));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __nv_bfloat16 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __nv_bfloat16() const {
        return __float2bfloat16_rz(float(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p double data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator double() const {
        return static_cast<double>(float(*this));
    }

    /* Convert to integral */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p char data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned char() const {
        unsigned char i;
        const float f = float(*this);

        if (f < 0.0f) {
            // saturate minimum
            i = 0U;
        } else {
            // normal value
            i = static_cast<unsigned char>(f);
        }
        return i;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p short \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned short int() const {
        return __half2ushort_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned int() const {
        return __half2uint_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p long \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned long int() const {
        unsigned long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<unsigned long>(__half2ull_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<unsigned long>(__half2uint_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p long \p long \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned long long int() const {
        return __half2ull_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p signed \p char data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator signed char() const {
        const float f = float(*this);
        return static_cast<signed char>(f);
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to an implementation defined \p char data type.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.
     * 
     * Clamps inputs to the output range.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator char() const {
        char value;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (((char)-1) < (char)0)
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            value = static_cast<char>(static_cast<signed char>(*this));
        }
        else
        {
            value = static_cast<char>(static_cast<unsigned char>(*this));
        }
        return value;
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p short \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator short int() const {
        return __half2short_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator int() const {
        return __half2int_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p long \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator long int() const {
        long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<long>(__half2ll_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<long>(__half2int_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p long \p long \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator long long int() const {
        return __half2ll_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator bool() const {
        return (__x & 0x1FU) != 0U;
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP6X2_E3M2_STRUCT C++ struct for handling vector type of two fp6 values of e3m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6X2_E3M2_STRUCT
 * \brief __nv_fp6x2_e3m2 datatype
 *
 * \details This structure implements the datatype for handling two
 * \p fp6 floating-point numbers of \p e3m2 kind each.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(2) __nv_fp6x2_e3m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6X2_E3M2_STRUCT
     * Storage variable contains the vector of two \p fp6 floating-point data
     * values.
     */
    __nv_fp6x2_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6x2_e3m2() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e3m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __half2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e3m2(const __half2 f) {
        __x = __nv_cvt_halfraw2_to_fp6x2(static_cast<__half2_raw>(f),
                                         __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __nv_bfloat162 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e3m2(const __nv_bfloat162 f) {
        __x = __nv_cvt_bfloat16raw2_to_fp6x2(static_cast<__nv_bfloat162_raw>(f),
                                             __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e3m2(const float2 f) {
        __x = __nv_cvt_float2_to_fp6x2(f, __NV_E3M2, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e3m2(const double2 f) {
        __x = __nv_cvt_double2_to_fp6x2(f, __NV_E3M2, cudaRoundNearest);
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening converts */
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __half2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __half2() const {
        return static_cast<__half2>(__nv_cvt_fp6x2_to_halfraw2(__x, __NV_E3M2));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float2() const {
        return __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(__x, __NV_E3M2));
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP6X4_E3M2_STRUCT C++ struct for handling vector type of four fp6 values of e3m2 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6X4_E3M2_STRUCT
 * \brief __nv_fp6x4_e3m2 datatype
 *
 * \details This structure implements the datatype for handling four
 * \p fp6 floating-point numbers of \p e3m2 kind each.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(4) __nv_fp6x4_e3m2 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6X4_E3M2_STRUCT
     * Storage variable contains the vector of four \p fp6 floating-point data
     * values.
     */
    __nv_fp6x4_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6x4_e3m2() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from a pair of \p __half2 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const __half2 flo,
                                                     const __half2 fhi) {
        const __nv_fp6x2_storage_t rlo = __nv_cvt_halfraw2_to_fp6x2(
            static_cast<__half2_raw>(flo), __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi = __nv_cvt_halfraw2_to_fp6x2(
            static_cast<__half2_raw>(fhi), __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from a pair of \p __nv_bfloat162 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const __nv_bfloat162 flo,
                                                     const __nv_bfloat162 fhi) {
        const __nv_fp6x2_storage_t rlo = __nv_cvt_bfloat16raw2_to_fp6x2(
            static_cast<__nv_bfloat162_raw>(flo), __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi = __nv_cvt_bfloat16raw2_to_fp6x2(
            static_cast<__nv_bfloat162_raw>(fhi), __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const float4 f) {
        const float2 flo = {f.x, f.y};
        const float2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_float2_to_fp6x2(flo, __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_float2_to_fp6x2(fhi, __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    __NV_SILENCE_DEPRECATION_BEGIN
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4 f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    __NV_SILENCE_DEPRECATION_END
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4_16a vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4_16a f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4_32a vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e3m2(const double4_32a f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E3M2, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E3M2, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening converts */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float4 vector data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float4() const {
        const __nv_fp6x2_storage_t slo = static_cast<__nv_fp6x2_storage_t>(__x);
        const __nv_fp6x2_storage_t shi =
            static_cast<__nv_fp6x2_storage_t>(__x >> 16U);
        float2 rlo = __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(slo, __NV_E3M2));
        float2 rhi = __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(shi, __NV_E3M2));
        float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
        return res;
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP6_E2M3_STRUCT C++ struct for handling fp6 data type of e2m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6_E2M3_STRUCT
 * \brief __nv_fp6_e2m3 datatype
 *
 * \details This structure implements the datatype for storing
 * \p fp6 floating-point numbers of \p e2m3 kind:
 * with 1 sign, 2 exponent, 1 implicit and 3 explicit mantissa bits.
 * This encoding does not support Inf/NaN.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(1) __nv_fp6_e2m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6_E2M3_STRUCT
     * Storage variable contains the \p fp6 floating-point data.
     */
    __nv_fp6_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6_e2m3() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider FP types */
    /* Note we do avoid constructor init-list because of special host/device
     * compilation rules */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __half data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const __half f) {
        __x = __nv_cvt_halfraw_to_fp6(static_cast<__half_raw>(f),
                                      __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __nv_bfloat16 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const __nv_bfloat16 f) {
        __x = __nv_cvt_bfloat16raw_to_fp6(static_cast<__nv_bfloat16_raw>(f),
                                          __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const float f) {
        __x = __nv_cvt_float_to_fp6(f, __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values and \p cudaRoundNearest rounding mode.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const double f) {
        __x = __nv_cvt_double_to_fp6(f, __NV_E2M3, cudaRoundNearest);
    }

    /* Converts from integral */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p short \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__
    __nv_fp6_e2m3(const unsigned short int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const unsigned int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const unsigned long int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p unsigned \p long \p long \p int data type, relies on
     * \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__
    __nv_fp6_e2m3(const unsigned long long int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p short \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const short int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p long \p int data type, relies on \p __NV_SATFINITE behavior
     * for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const long int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p long \p long \p int data type, relies on \p
     * __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6_e2m3(const long long int val) {
        __x = static_cast<__nv_fp6_e2m3>(static_cast<float>(val)).__x;
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening FP converts */
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __half data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __half() const {
        return static_cast<__half>(__nv_cvt_fp6_to_halfraw(__x, __NV_E2M3));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float() const {
        return __internal_halfraw_to_float(
            __nv_cvt_fp6_to_halfraw(__x, __NV_E2M3));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __nv_bfloat16 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __nv_bfloat16() const {
        return __float2bfloat16_rz(float(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p double data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator double() const {
        return static_cast<double>(float(*this));
    }

    /* Convert to integral */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p char data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned char() const {
        unsigned char i;
        const float f = float(*this);

        if (f < 0.0f) {
            // saturate minimum
            i = 0U;
        } else {
            // normal value
            i = static_cast<unsigned char>(f);
        }
        return i;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p short \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned short int() const {
        return __half2ushort_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned int() const {
        return __half2uint_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p long \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned long int() const {
        unsigned long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<unsigned long>(__half2ull_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<unsigned long>(__half2uint_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p unsigned \p long \p long \p int data type.
     * Clamps negative inputs to zero.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator unsigned long long int() const {
        return __half2ull_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p signed \p char data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator signed char() const {
        const float f = float(*this);
        return static_cast<signed char>(f);
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to an implementation defined \p char data type.
     * 
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.
     * 
     * Clamps inputs to the output range.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator char() const {
        char value;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (((char)-1) < (char)0)
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            value = static_cast<char>(static_cast<signed char>(*this));
        }
        else
        {
            value = static_cast<char>(static_cast<unsigned char>(*this));
        }
        return value;
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p short \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator short int() const {
        return __half2short_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator int() const {
        return __half2int_rz(__half(*this));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p long \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator long int() const {
        long retval;
        /* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (push)
#pragma warning (disable: 4127)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER) && !defined(__CUDA_ARCH__)
#pragma warning (pop)
#endif /* _MSC_VER && !defined(__CUDA_ARCH__) */
        {
            retval = static_cast<long>(__half2ll_rz(__half(*this)));
        }
        else
        {
            retval = static_cast<long>(__half2int_rz(__half(*this)));
        }
        return retval;
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p long \p long \p int data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator long long int() const {
        return __half2ll_rz(__half(*this));
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator bool() const {
        return (__x & 0x1FU) != 0U;
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP6X2_E2M3_STRUCT C++ struct for handling vector type of two fp6 values of e2m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6X2_E2M3_STRUCT
 * \brief __nv_fp6x2_e2m3 datatype
 *
 * \details This structure implements the datatype for handling two
 * \p fp6 floating-point numbers of \p e2m3 kind each.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(2) __nv_fp6x2_e2m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6X2_E2M3_STRUCT
     * Storage variable contains the vector of two \p fp6 floating-point data
     * values.
     */
    __nv_fp6x2_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6x2_e2m3() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e2m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __half2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e2m3(const __half2 f) {
        __x = __nv_cvt_halfraw2_to_fp6x2(static_cast<__half2_raw>(f),
                                         __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p __nv_bfloat162 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e2m3(const __nv_bfloat162 f) {
        __x = __nv_cvt_bfloat16raw2_to_fp6x2(static_cast<__nv_bfloat162_raw>(f),
                                             __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e2m3(const float2 f) {
        __x = __nv_cvt_float2_to_fp6x2(f, __NV_E2M3, cudaRoundNearest);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double2 data type, relies on \p __NV_SATFINITE
     * behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x2_e2m3(const double2 f) {
        __x = __nv_cvt_double2_to_fp6x2(f, __NV_E2M3, cudaRoundNearest);
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening converts */
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p __half2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator __half2() const {
        return static_cast<__half2>(__nv_cvt_fp6x2_to_halfraw2(__x, __NV_E2M3));
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float2 data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float2() const {
        return __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(__x, __NV_E2M3));
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

/**
 * \defgroup CUDA_MATH_FP6X4_E2M3_STRUCT C++ struct for handling vector type of four fp6 values of e2m3 kind.
 * \ingroup CUDA_MATH_INTRINSIC_FP6
 */

/**
 * \ingroup CUDA_MATH_FP6X4_E2M3_STRUCT
 * \brief __nv_fp6x4_e2m3 datatype
 *
 * \details This structure implements the datatype for handling four
 * \p fp6 floating-point numbers of \p e2m3 kind each.
 *
 * The structure implements converting constructors and operators.
 */
struct __CUDA_ALIGN__(4) __nv_fp6x4_e2m3 {
  public:
    /**
     * \ingroup CUDA_MATH_FP6X4_E2M3_STRUCT
     * Storage variable contains the vector of four \p fp6 floating-point data
     * values.
     */
    __nv_fp6x4_storage_t __x;

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP6)
    __nv_fp6x4_e2m3() = default;
#else
    __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3() {}
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP6) */

#if !defined(__CUDA_NO_FP6_CONVERSIONS__)

    /* Construct from wider types */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from a pair of \p __half2 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const __half2 flo,
                                                     const __half2 fhi) {
        const __nv_fp6x2_storage_t rlo = __nv_cvt_halfraw2_to_fp6x2(
            static_cast<__half2_raw>(flo), __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi = __nv_cvt_halfraw2_to_fp6x2(
            static_cast<__half2_raw>(fhi), __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from a pair of \p __nv_bfloat162 data type values,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const __nv_bfloat162 flo,
                                                     const __nv_bfloat162 fhi) {
        const __nv_fp6x2_storage_t rlo = __nv_cvt_bfloat16raw2_to_fp6x2(
            static_cast<__nv_bfloat162_raw>(flo), __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi = __nv_cvt_bfloat16raw2_to_fp6x2(
            static_cast<__nv_bfloat162_raw>(fhi), __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p float4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const float4 f) {
        const float2 flo = {f.x, f.y};
        const float2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_float2_to_fp6x2(flo, __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_float2_to_fp6x2(fhi, __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4 vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    __NV_SILENCE_DEPRECATION_BEGIN
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4 f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }
    __NV_SILENCE_DEPRECATION_END
    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4_16a vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4_16a f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Constructor from \p double4_32a vector data type,
     * relies on \p __NV_SATFINITE behavior for out-of-range values.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ __nv_fp6x4_e2m3(const double4_32a f) {
        const double2 flo = {f.x, f.y};
        const double2 fhi = {f.z, f.w};
        const __nv_fp6x2_storage_t rlo =
            __nv_cvt_double2_to_fp6x2(flo, __NV_E2M3, cudaRoundNearest);
        const __nv_fp6x2_storage_t rhi =
            __nv_cvt_double2_to_fp6x2(fhi, __NV_E2M3, cudaRoundNearest);
        __x = __internal_pack_u16x2_to_u32(rlo, rhi);
    }

#if !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__)
    /* Widening converts */

    /**
     * \ingroup CUDA_MATH_FP6_MISC
     * Conversion operator to \p float4 vector data type.
     */
    explicit __CUDA_HOSTDEVICE_FP6__ operator float4() const {
        const __nv_fp6x2_storage_t slo = static_cast<__nv_fp6x2_storage_t>(__x);
        const __nv_fp6x2_storage_t shi =
            static_cast<__nv_fp6x2_storage_t>(__x >> 16U);
        float2 rlo = __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(slo, __NV_E2M3));
        float2 rhi = __internal_halfraw2_to_float2(
            __nv_cvt_fp6x2_to_halfraw2(shi, __NV_E2M3));
        float4 res = {rlo.x, rlo.y, rhi.x, rhi.y};
        return res;
    }
#endif /* !defined(__CUDA_NO_FP6_CONVERSION_OPERATORS__) */
#endif /* !defined(__CUDA_NO_FP6_CONVERSIONS__) */
};

#endif /* defined(__cplusplus) */

#endif /* end of include guard: __CUDA_FP6_HPP__ */
