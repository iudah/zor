#ifndef _ZOR_SIMD_DEFINES_H_
#define _ZOR_SIMD_DEFINES_H_


#if defined(__ARM_NEON__) || defined(__AVX__) || defined(__SSE__)
#if  defined(__ARM_NEON__) && defined(__ARM_FP) && (__ARM_FP & 2)
#define use_float16
#elif defined(__ARM_NEON__) && defined(__ARM_FP) && defined(__ARM_FEATURE_BF16)
#define use_bfp16
#else
#define use_simd_float32
#endif
#elif defined(__FLT16_MAX__)
#define use_gnu_float16
#else
#define use_float32
#endif

#if defined(use_float16)
typedef __fp16 zfl;
#elif defined(use_bfp16)
typedef __bf16 zfl;
#elif defined(use_simd_float32)
typedef float zfl;
#elif defined(use_gnu_float16)
typedef _Float16 zfl;
#elif defined(use_float32)
typedef float zfl;
#endif





#endif // _ZOR_SIMD_DEFINES_H_