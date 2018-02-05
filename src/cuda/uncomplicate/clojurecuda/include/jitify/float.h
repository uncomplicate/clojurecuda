#ifndef _float_h_
#define _float_h_

inline __host__ __device__ float  jitify_int_as_float(int i) {
    union FloatInt { float f; int i; } fi;
    fi.i = i;
    return fi.f;
}

inline __host__ __device__ double jitify_longlong_as_double(long long i) {
    union DoubleLongLong { double f; long long i; } fi;
    fi.i = i;
    return fi.f;
}

#define FLT_RADIX       2
#define FLT_MANT_DIG    24
#define DBL_MANT_DIG    53
#define FLT_DIG         6
#define DBL_DIG         15
#define FLT_MIN_EXP     -125
#define DBL_MIN_EXP     -1021
#define FLT_MIN_10_EXP  -37
#define DBL_MIN_10_EXP  -307
#define FLT_MAX_EXP     128
#define DBL_MAX_EXP     1024
#define FLT_MAX_10_EXP  38
#define DBL_MAX_10_EXP  308
#define FLT_MAX         jitify_int_as_float(2139095039)
#define DBL_MAX         jitify_longlong_as_double(9218868437227405311)
#define FLT_EPSILON     jitify_int_as_float(872415232)
#define DBL_EPSILON     jitify_longlong_as_double(4372995238176751616)
#define FLT_MIN         jitify_int_as_float(8388608)
#define DBL_MIN         jitify_longlong_as_double(4503599627370496)
#define FLT_ROUNDS      1

#endif
