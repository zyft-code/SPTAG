// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/DistanceUtils.h"
#include <immintrin.h>

using namespace SPTAG;
using namespace SPTAG::COMMON;

#ifndef _MSC_VER
#define DIFF128 diff128
#define DIFF256 diff256
#else
#define DIFF128 diff128.m128_f32
#define DIFF256 diff256.m256_f32
#endif

DistanceUtils::DistanceUtils()
{
	InstructionSet::AVX512();
}

if_SSE(inline __m128 _mm_mul_epi8)(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

if_SSE(inline __m128 _mm_sqdf_epi8)(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi8(X, zero);
    __m128i sign_y = _mm_cmplt_epi8(Y, zero);

    __m128i xlo = _mm_unpacklo_epi8(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi8(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi8(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi8(Y, sign_y);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

if_SSE(inline __m128 _mm_mul_epu8)(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(xlo, ylo), _mm_madd_epi16(xhi, yhi)));
}

if_SSE(inline __m128 _mm_sqdf_epu8)(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i xlo = _mm_unpacklo_epi8(X, zero);
    __m128i xhi = _mm_unpackhi_epi8(X, zero);
    __m128i ylo = _mm_unpacklo_epi8(Y, zero);
    __m128i yhi = _mm_unpackhi_epi8(Y, zero);

    __m128i dlo = _mm_sub_epi16(xlo, ylo);
    __m128i dhi = _mm_sub_epi16(xhi, yhi);

    return _mm_cvtepi32_ps(_mm_add_epi32(_mm_madd_epi16(dlo, dlo), _mm_madd_epi16(dhi, dhi)));
}

if_SSE(inline __m128 _mm_mul_epi16)(__m128i X, __m128i Y)
{
    return _mm_cvtepi32_ps(_mm_madd_epi16(X, Y));
}

if_SSE(inline __m128 _mm_sqdf_epi16)(__m128i X, __m128i Y)
{
    __m128i zero = _mm_setzero_si128();

    __m128i sign_x = _mm_cmplt_epi16(X, zero);
    __m128i sign_y = _mm_cmplt_epi16(Y, zero);

    __m128i xlo = _mm_unpacklo_epi16(X, sign_x);
    __m128i xhi = _mm_unpackhi_epi16(X, sign_x);
    __m128i ylo = _mm_unpacklo_epi16(Y, sign_y);
    __m128i yhi = _mm_unpackhi_epi16(Y, sign_y);

    __m128 dlo = _mm_cvtepi32_ps(_mm_sub_epi32(xlo, ylo));
    __m128 dhi = _mm_cvtepi32_ps(_mm_sub_epi32(xhi, yhi));

    return _mm_add_ps(_mm_mul_ps(dlo, dlo), _mm_mul_ps(dhi, dhi));
}

if_SSE(inline __m128 _mm_sqdf_ps)(__m128 X, __m128 Y)
{
    __m128 d = _mm_sub_ps(X, Y);
    return _mm_mul_ps(d, d);
}

if_AVX2(inline __m256 _mm256_mul_epi8)(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

f_AVX2(__m256 _mm256_sqdf_epi8)(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi8(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi8(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi8(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi8(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi8(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi8(Y, sign_y);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

if_AVX2(inline __m256 _mm256_mul_epu8)(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(xlo, ylo), _mm256_madd_epi16(xhi, yhi)));
}

if_AVX2(inline __m256 _mm256_sqdf_epu8)(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i xlo = _mm256_unpacklo_epi8(X, zero);
    __m256i xhi = _mm256_unpackhi_epi8(X, zero);
    __m256i ylo = _mm256_unpacklo_epi8(Y, zero);
    __m256i yhi = _mm256_unpackhi_epi8(Y, zero);

    __m256i dlo = _mm256_sub_epi16(xlo, ylo);
    __m256i dhi = _mm256_sub_epi16(xhi, yhi);

    return _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_madd_epi16(dlo, dlo), _mm256_madd_epi16(dhi, dhi)));
}

if_AVX2(inline __m256 _mm256_mul_epi16)(__m256i X, __m256i Y)
{
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(X, Y));
}

if_AVX2(inline __m256 _mm256_sqdf_epi16)(__m256i X, __m256i Y)
{
    __m256i zero = _mm256_setzero_si256();

    __m256i sign_x = _mm256_cmpgt_epi16(zero, X);
    __m256i sign_y = _mm256_cmpgt_epi16(zero, Y);

    __m256i xlo = _mm256_unpacklo_epi16(X, sign_x);
    __m256i xhi = _mm256_unpackhi_epi16(X, sign_x);
    __m256i ylo = _mm256_unpacklo_epi16(Y, sign_y);
    __m256i yhi = _mm256_unpackhi_epi16(Y, sign_y);

    __m256 dlo = _mm256_cvtepi32_ps(_mm256_sub_epi32(xlo, ylo));
    __m256 dhi = _mm256_cvtepi32_ps(_mm256_sub_epi32(xhi, yhi));

    return _mm256_add_ps(_mm256_mul_ps(dlo, dlo), _mm256_mul_ps(dhi, dhi));
}

if_AVX(inline __m256 _mm256_sqdf_ps)(__m256 X, __m256 Y)
{
    __m256 d = _mm256_sub_ps(X, Y);
    return _mm256_mul_ps(d, d);
}

// Do not use intrinsics not supported by old MS compiler version
#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
if_AVX512(inline __m512 _mm512_mul_epi8)(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

if_AVX512(inline __m512 _mm512_sqdf_epi8)(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask64 sign_x_mask = _mm512_cmpgt_epi8_mask (zero, X);
    __mmask64 sign_y_mask = _mm512_cmpgt_epi8_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi8(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi8(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi8(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi8(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi8(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi8(Y, sign_y);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

if_AVX512(inline __m512 _mm512_mul_epu8)(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

if_AVX512(inline __m512 _mm512_sqdf_epu8)(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    __m512i dlo = _mm512_sub_epi16(xlo, ylo);
    __m512i dhi = _mm512_sub_epi16(xhi, yhi);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(dlo, dlo), _mm512_madd_epi16(dhi, dhi)));
}

if_AVX512(inline __m512 _mm512_mul_epi16)(__m512i X, __m512i Y)
{
    return _mm512_cvtepi32_ps(_mm512_madd_epi16(X, Y));
}

if_AVX512(inline __m512 _mm512_sqdf_epi16)(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __mmask32 sign_x_mask = _mm512_cmpgt_epi16_mask (zero, X);
    __mmask32 sign_y_mask = _mm512_cmpgt_epi16_mask (zero, Y);

    __m512i sign_x = _mm512_movm_epi16(sign_x_mask);
    __m512i sign_y = _mm512_movm_epi16(sign_y_mask);

    __m512i xlo = _mm512_unpacklo_epi16(X, sign_x);
    __m512i xhi = _mm512_unpackhi_epi16(X, sign_x);
    __m512i ylo = _mm512_unpacklo_epi16(Y, sign_y);
    __m512i yhi = _mm512_unpackhi_epi16(Y, sign_y);

    __m512 dlo = _mm512_cvtepi32_ps(_mm512_sub_epi32(xlo, ylo));
    __m512 dhi = _mm512_cvtepi32_ps(_mm512_sub_epi32(xhi, yhi));

    return _mm512_add_ps(_mm512_mul_ps(dlo, dlo), _mm512_mul_ps(dhi, dhi));
}

if_AVX512(inline __m512 _mm512_sqdf_ps)(__m512 X, __m512 Y)
{
    __m512 d = _mm512_sub_ps(X, Y);
    return _mm512_mul_ps(d, d);
}
#endif


#define REPEAT(type, ctype, delta, load, exec, acc, result) \
            { \
                type c1 = load((ctype *)(pX)); \
                type c2 = load((ctype *)(pY)); \
                pX += delta; pY += delta; \
                result = acc(result, exec(c1, c2)); \
            } \

