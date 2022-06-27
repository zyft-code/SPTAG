// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "inc/Core/Common/DistanceUtils.h"
#include "inc/Core/Common/DistanceUtilsSSE.h"
#include "inc/Core/Common/DistanceUtilsAVX.h"
#include "inc/Core/Common/DistanceUtilsAVX512.h"
#include <immintrin.h>

using namespace SPTAG;
using namespace SPTAG::COMMON;

float DistanceUtils::ComputeL2Distance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_sqdf_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_sqdf_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_sqdf_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_sqdf_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_sqdf_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_sqdf_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
        c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }

    while (pX < pEnd1) {
        float c1 = ((float)(*pX++) - (float)(*pY++)); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeL2Distance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_sqdf_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_sqdf_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_sqdf_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) {
        float c1 = (*pX++) - (*pY++); diff += c1 * c1;
    }
    return diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int8_t* pX, const std::int8_t* pY, DimensionType length)
{
    const std::int8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::int8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epi8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epi8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epi8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 16129 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::uint8_t* pX, const std::uint8_t* pY, DimensionType length)
{
    const std::uint8_t* pEnd32 = pX + ((length >> 5) << 5);
    const std::uint8_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::uint8_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::uint8_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::uint8_t* pEnd64 = pX + ((length >> 6) << 6);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd64) {
        REPEAT(__m512i, __m512i, 64, _mm512_loadu_si512, _mm512_mul_epu8, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd32) {
        REPEAT(__m256i, __m256i, 32, _mm256_loadu_si256, _mm256_mul_epu8, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd16) {
        REPEAT(__m128i, __m128i, 16, _mm_loadu_si128, _mm_mul_epu8, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }
    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return 65025 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const std::int16_t* pX, const std::int16_t* pY, DimensionType length)
{
    const std::int16_t* pEnd16 = pX + ((length >> 4) << 4);
    const std::int16_t* pEnd8 = pX + ((length >> 3) << 3);
    const std::int16_t* pEnd4 = pX + ((length >> 2) << 2);
    const std::int16_t* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const std::int16_t* pEnd32 = pX + ((length >> 5) << 5);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd32) {
        REPEAT(__m512i, __m512i, 32, _mm512_loadu_si512, _mm512_mul_epi16, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd16) {
        REPEAT(__m256i, __m256i, 16, _mm256_loadu_si256, _mm256_mul_epi16, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd8) {
        REPEAT(__m128i, __m128i, 8, _mm_loadu_si128, _mm_mul_epi16, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd4)
    {
        float c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
        c1 = ((float)(*pX++) * (float)(*pY++)); diff += c1;
    }

    while (pX < pEnd1) diff += ((float)(*pX++) * (float)(*pY++));
    return  1073676289 - diff;
}

float DistanceUtils::ComputeCosineDistance_AVX512(const float* pX, const float* pY, DimensionType length)
{
    const float* pEnd8 = pX + ((length >> 3) << 3);
    const float* pEnd4 = pX + ((length >> 2) << 2);
    const float* pEnd1 = pX + length;

#if (!defined _MSC_VER) || (_MSC_VER >= 1920)
    const float* pEnd16 = pX + ((length >> 4) << 4);
    __m512 diff512 = _mm512_setzero_ps();
    while (pX < pEnd16) {
        REPEAT(__m512, const float, 16, _mm512_loadu_ps, _mm512_mul_ps, _mm512_add_ps, diff512)
    }
    __m256 diff256 = _mm256_add_ps(_mm512_castps512_ps256(diff512), _mm512_extractf32x8_ps(diff512, 1));
#else
    __m256 diff256 = _mm256_setzero_ps();
#endif

    while (pX < pEnd8)
    {
        REPEAT(__m256, const float, 8, _mm256_loadu_ps, _mm256_mul_ps, _mm256_add_ps, diff256)
    }
    __m128 diff128 = _mm_add_ps(_mm256_castps256_ps128(diff256), _mm256_extractf128_ps(diff256, 1));
    while (pX < pEnd4)
    {
        REPEAT(__m128, const float, 4, _mm_loadu_ps, _mm_mul_ps, _mm_add_ps, diff128)
    }
    float diff = DIFF128[0] + DIFF128[1] + DIFF128[2] + DIFF128[3];

    while (pX < pEnd1) diff += (*pX++) * (*pY++);
    return 1 - diff;
}

