#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef ALIASES_HPP
#define ALIASES_HPP

#ifdef MATTY_USE_x86_SIMD_128
#include <x86intrin.h>

namespace {
template <bool cond, auto value1, auto value2>
struct conditional_value;

template <auto value1, auto value2>
struct conditional_value<true, value1, value2> {
    constexpr static auto value = value1;
};

template <auto value1, auto value2>
struct conditional_value<false, value1, value2> {
    constexpr static auto value = value2;
};

template <bool cond, auto value1, auto value2>
constexpr auto conditional_value_v = conditional_value<cond, value1, value2>::value;
} // namespace

template <typename T>
struct Aliases {
    static_assert(CHAR_BIT == 8 && "Only 8-bit bytes supported");
    static_assert(sizeof(T) == 4, "Only 32-bit types supported");
    static_assert(
        std::is_integral_v<T> || std::is_floating_point_v<T>, "Only integral or floating point types supported");

    using integral_scalar_32bit = __m128i (*)(__m128i a, __m128i b);
    using floating_scalar_32bit = __m128 (*)(__m128 a, __m128 b);

    using integral_shuffle_32bit = __m128i (*)(__m128i a, int imm8);
    using floating_shuffle_32bit = __m128 (*)(__m128 a, int imm8);

    using load_128bit_integral = __m128i (*)(const __m128i *ptr);
    using load_128bit_floating = __m128 (*)(const float *ptr);

    using store_128bit_integral = void (*)(__m128i *ptr, __m128i a);
    using store_128bit_floating = void (*)(float *ptr, __m128 a);

    using load_128bit_op = std::conditional_t<std::is_integral_v<T>, load_128bit_integral, load_128bit_floating>;
    using store_128bit_op = std::conditional_t<std::is_integral_v<T>, store_128bit_integral, store_128bit_floating>;

    using scalar_32bit_op = std::conditional_t<std::is_integral_v<T>, integral_scalar_32bit, floating_scalar_32bit>;
    using shuffle_32bit_op = std::conditional_t<std::is_integral_v<T>, integral_shuffle_32bit, floating_shuffle_32bit>;

    using value_128bit = std::conditional_t<std::is_integral_v<T>, __m128i, __m128>;

    using set1_32bit_integral = __m128i (*)(T a);
    using set1_32bit_floating = __m128 (*)(T a);
    using set1_32bit_op = std::conditional_t<std::is_integral_v<T>, set1_32bit_integral, set1_32bit_floating>;

    using set_32bit_integral = __m128i (*)(T a, T b, T c, T d);
    using set_32bit_floating = __m128 (*)(T a, T b, T c, T d);
    using set_32bit_op = std::conditional_t<std::is_integral_v<T>, set_32bit_integral, set_32bit_floating>;

    using load_128bit_value = std::conditional_t<std::is_integral_v<T>, const __m128i *, const float *>;
    using store_128bit_value = std::conditional_t<std::is_integral_v<T>, __m128i *, float *>;

    static inline __m128i multiply_32_integral(__m128i a, __m128i b) {
#ifdef __SSE4_1__ // modern CPU - use SSE 4.1
        return _mm_mullo_epi32(a, b);
#else  // old CPU - use SSE 2
        __m128i tmp1 = _mm_mul_epu32(a, b);                                       /* mul 2,0*/
        __m128i tmp2 = _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1 */
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
            _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))); /* shuffle results to [63..0] and pack */
#endif // __SSE4_1__
    }

    static inline __m128i add_32_integral(__m128i a, __m128i b) { return _mm_add_epi32(a, b); }
    static inline __m128 add_32_floating(__m128 a, __m128 b) { return _mm_add_ps(a, b); }
    static inline __m128i sub_32_integral(__m128i a, __m128i b) { return _mm_sub_epi32(a, b); }
    static inline __m128 sub_32_floating(__m128 a, __m128 b) { return _mm_sub_ps(a, b); }

    static inline __m128i shuffle_32_integral(__m128i a, int imm8) { return _mm_shuffle_epi32(a, imm8); }
    static inline __m128 shuffle_32_floating(__m128 a, int imm8) { return _mm_shuffle_ps(a, a, imm8); }

    static inline __m128i hadd_32_integral(__m128i a, __m128i b) { return _mm_hadd_epi32(a, b); }
    static inline __m128 hadd_32_floating(__m128 a, __m128 b) { return _mm_hadd_ps(a, b); }

    static inline __m128i set_32_integral(T a, T b, T c, T d) { return _mm_set_epi32(d, c, b, a); }
    static inline __m128 set_32_floating(T a, T b, T c, T d) { return _mm_set_ps(d, c, b, a); }
    static inline __m128i set1_32_integral(T a) { return _mm_set1_epi32(a); }
    static inline __m128 set1_32_floating(T a) { return _mm_set1_ps(a); }

    static inline __m128i load_32_integral(const __m128i *ptr) { return _mm_load_si128(ptr); }
    static inline __m128 load_32_floating(const float *ptr) { return _mm_load_ps(ptr); }
    static inline void store_32_integral(__m128i *ptr, __m128i a) { _mm_store_si128(ptr, a); }
    static inline void store_32_floating(float *ptr, __m128 a) { _mm_store_ps(ptr, a); }

    static inline __m128 multiply_32_floating(__m128 a, __m128 b) { return _mm_mul_ps(a, b); }

    constexpr static scalar_32bit_op add_32bit =
        conditional_value_v<std::is_integral_v<T>, add_32_integral, add_32_floating>;
    constexpr static scalar_32bit_op sub_32bit =
        conditional_value_v<std::is_integral_v<T>, sub_32_integral, sub_32_floating>;

    constexpr static scalar_32bit_op mul_32bit =
        conditional_value_v<std::is_integral_v<T>, multiply_32_integral, multiply_32_floating>;
    constexpr static scalar_32bit_op hadd_32bit =
        conditional_value_v<std::is_integral_v<T>, hadd_32_integral, hadd_32_floating>;

    constexpr static shuffle_32bit_op shuffle_32bit =
        conditional_value_v<std::is_integral_v<T>, shuffle_32_integral, shuffle_32_floating>;

    constexpr static load_128bit_op load_128bit =
        conditional_value_v<std::is_integral_v<T>, load_32_integral, load_32_floating>;
    constexpr static store_128bit_op store_128bit =
        conditional_value_v<std::is_integral_v<T>, store_32_integral, store_32_floating>;

    constexpr static set_32bit_op set_32bit =
        conditional_value_v<std::is_integral_v<T>, set_32_integral, set_32_floating>;
    constexpr static set1_32bit_op set1_32bit =
        conditional_value_v<std::is_integral_v<T>, set1_32_integral, set1_32_floating>;
};

#define CAST_LOAD_128BIT(x)  reinterpret_cast<load_128bit_value>(x)
#define CAST_STORE_128BIT(x) reinterpret_cast<store_128bit_value>(x)

#define USING_ALL_ALIASES                                                                                              \
    using integral_scalar_32bit = typename Aliases<T>::integral_scalar_32bit;                                          \
    using floating_scalar_32bit = typename Aliases<T>::floating_scalar_32bit;                                          \
                                                                                                                       \
    using integral_shuffle_32bit = typename Aliases<T>::integral_shuffle_32bit;                                        \
    using floating_shuffle_32bit = typename Aliases<T>::floating_shuffle_32bit;                                        \
                                                                                                                       \
    using load_128bit_integral = typename Aliases<T>::load_128bit_integral;                                            \
    using store_128bit_integral = typename Aliases<T>::store_128bit_integral;                                          \
                                                                                                                       \
    using load_128bit_floating = typename Aliases<T>::load_128bit_floating;                                            \
    using store_128bit_floating = typename Aliases<T>::store_128bit_floating;                                          \
                                                                                                                       \
    using load_128bit_op = typename Aliases<T>::load_128bit_op;                                                        \
    using store_128bit_op = typename Aliases<T>::store_128bit_op;                                                      \
                                                                                                                       \
    using scalar_32bit_op = typename Aliases<T>::scalar_32bit_op;                                                      \
    using shuffle_32bit_op = typename Aliases<T>::shuffle_32bit_op;                                                    \
                                                                                                                       \
    using value_128bit = typename Aliases<T>::value_128bit;                                                            \
                                                                                                                       \
    using set1_32bit_integral = typename Aliases<T>::set1_32bit_integral;                                              \
    using set1_32bit_floating = typename Aliases<T>::set1_32bit_floating;                                              \
    using set1_32bit_op = typename Aliases<T>::set1_32bit_op;                                                          \
                                                                                                                       \
    using set_32bit_integral = typename Aliases<T>::set_32bit_integral;                                                \
    using set_32bit_floating = typename Aliases<T>::set_32bit_floating;                                                \
    using set_32bit_op = typename Aliases<T>::set_32bit_op;                                                            \
                                                                                                                       \
    using load_128bit_value = typename Aliases<T>::load_128bit_value;                                                  \
    using store_128bit_value = typename Aliases<T>::store_128bit_value;                                                \
                                                                                                                       \
    using Aliases<T>::add_32_integral;                                                                                 \
    using Aliases<T>::add_32_floating;                                                                                 \
    using Aliases<T>::sub_32_integral;                                                                                 \
    using Aliases<T>::sub_32_floating;                                                                                 \
                                                                                                                       \
    using Aliases<T>::shuffle_32_integral;                                                                             \
    using Aliases<T>::shuffle_32_floating;                                                                             \
                                                                                                                       \
    using Aliases<T>::hadd_32_integral;                                                                                \
    using Aliases<T>::hadd_32_floating;                                                                                \
                                                                                                                       \
    using Aliases<T>::set_32_integral;                                                                                 \
    using Aliases<T>::set_32_floating;                                                                                 \
                                                                                                                       \
    using Aliases<T>::set1_32_integral;                                                                                \
    using Aliases<T>::set1_32_floating;                                                                                \
                                                                                                                       \
    using Aliases<T>::load_32_integral;                                                                                \
    using Aliases<T>::load_32_floating;                                                                                \
    using Aliases<T>::store_32_integral;                                                                               \
    using Aliases<T>::store_32_floating;                                                                               \
                                                                                                                       \
    using Aliases<T>::multiply_32_floating;                                                                            \
                                                                                                                       \
    using Aliases<T>::add_32bit;                                                                                       \
    using Aliases<T>::sub_32bit;                                                                                       \
                                                                                                                       \
    using Aliases<T>::mul_32bit;                                                                                       \
    using Aliases<T>::hadd_32bit;                                                                                      \
                                                                                                                       \
    using Aliases<T>::shuffle_32bit;                                                                                   \
                                                                                                                       \
    using Aliases<T>::load_128bit;                                                                                     \
    using Aliases<T>::store_128bit;                                                                                    \
                                                                                                                       \
    using Aliases<T>::set_32bit;                                                                                       \
    using Aliases<T>::set1_32bit;

#endif // MATTY_USE_x86_SIMD_128

#endif // ALIASES_HPP