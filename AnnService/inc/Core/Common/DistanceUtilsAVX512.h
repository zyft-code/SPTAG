// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/DistanceUtils.h"
#include <immintrin.h>

using namespace SPTAG;
using namespace SPTAG::COMMON;

// Do not use intrinsics not supported by old MS compiler version
#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
inline __m512 _mm512_mul_epi8(__m512i X, __m512i Y)
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

inline __m512 _mm512_sqdf_epi8(__m512i X, __m512i Y)
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

inline __m512 _mm512_mul_epu8(__m512i X, __m512i Y)
{
    __m512i zero = _mm512_setzero_si512();

    __m512i xlo = _mm512_unpacklo_epi8(X, zero);
    __m512i xhi = _mm512_unpackhi_epi8(X, zero);
    __m512i ylo = _mm512_unpacklo_epi8(Y, zero);
    __m512i yhi = _mm512_unpackhi_epi8(Y, zero);

    return _mm512_cvtepi32_ps(_mm512_add_epi32(_mm512_madd_epi16(xlo, ylo), _mm512_madd_epi16(xhi, yhi)));
}

inline __m512 _mm512_sqdf_epu8(__m512i X, __m512i Y)
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

inline __m512 _mm512_mul_epi16(__m512i X, __m512i Y)
{
    return _mm512_cvtepi32_ps(_mm512_madd_epi16(X, Y));
}

inline __m512 _mm512_sqdf_epi16(__m512i X, __m512i Y)
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

inline __m512 _mm512_sqdf_ps(__m512 X, __m512 Y)
{
    __m512 d = _mm512_sub_ps(X, Y);
    return _mm512_mul_ps(d, d);
}
#endif