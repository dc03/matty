#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef MATTY_FIXED_MATRIX_4_HPP
#define MATTY_FIXED_MATRIX_4_HPP

#include "FixedMatrix.hpp"

#include <climits>
#include <type_traits>

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

namespace matty {
template <typename T>
class FixedMatrix4 {
    constexpr static std::size_t M = 4;
    constexpr static std::size_t N = 4;

#ifdef MATTY_USE_x86_SIMD_128
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

#define CAST_LOAD_128BIT(x)  reinterpret_cast<load_128bit_value>(x)
#define CAST_STORE_128BIT(x) reinterpret_cast<store_128bit_value>(x)

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

#endif // MATTY_USE_x86_SIMD_128

    constexpr std::size_t from_xy_index(std::size_t x, std::size_t y) { return x * 4 + y; }

    T data[M * N];

    template <typename U>
    constexpr void construct_with_one(U value);

  public:
    constexpr FixedMatrix4() noexcept;

    template <typename... Ts, typename = std::enable_if_t<(std::is_convertible_v<Ts, T> && ...)>>
    constexpr explicit FixedMatrix4(Ts &&...args) noexcept;

    constexpr FixedMatrix4(const FixedMatrix4<T> &other) noexcept;
    constexpr FixedMatrix4<T> &operator=(const FixedMatrix4<T> &other);

    constexpr FixedMatrix4(FixedMatrix4<T> &&other) noexcept;
    constexpr FixedMatrix4<T> &operator=(FixedMatrix4<T> &&other) noexcept;

    constexpr static FixedMatrix4<T> identity() noexcept;
    constexpr static FixedMatrix4<T> zero() noexcept;
    constexpr static FixedMatrix4<T> one() noexcept;

    constexpr FixedMatrix4<T> copy() const;
    constexpr T *operator[](std::size_t x);
    constexpr const T *operator[](std::size_t x) const noexcept;

    constexpr FixedMatrix4<T> add(const FixedMatrix4<T> &other) const;
    constexpr FixedMatrix4<T> &add_self(const FixedMatrix4<T> &other);

    constexpr FixedMatrix4<T> add(T scalar) const;
    constexpr FixedMatrix4<T> &add_self(T scalar);

    constexpr FixedMatrix4<T> sub(const FixedMatrix4<T> &other) const;
    constexpr FixedMatrix4<T> &sub_self(const FixedMatrix4<T> &other);

    constexpr FixedMatrix4<T> sub(T scalar) const;
    constexpr FixedMatrix4<T> &sub_self(T scalar);

    template <typename T2 = int>
    constexpr FixedMatrix4<T> mul(const FixedMatrix4<T2> &other) const;
    template <typename T2 = int>
    constexpr FixedMatrix4<T> &mul_self(const FixedMatrix4<T2> &other);

    constexpr FixedMatrix4<T> mul(T scalar) const;
    constexpr FixedMatrix4<T> &mul_self(T scalar);

    constexpr FixedMatrix4<T> div(T scalar) const;
    constexpr FixedMatrix4<T> &div_self(T scalar);

    constexpr FixedMatrix4<T> transpose() const;
    constexpr FixedMatrix4<T> &transpose_self();

    constexpr FixedMatrix4<T> horizontal_flip() const;
    constexpr FixedMatrix4<T> &horizontal_flip_self();

    constexpr FixedMatrix4<T> vertical_flip() const;
    constexpr FixedMatrix4<T> &vertical_flip_self();

    [[nodiscard]] constexpr std::size_t rows() const noexcept;
    [[nodiscard]] constexpr std::size_t columns() const noexcept;

    FixedMatrix4<T> operator+(const FixedMatrix4<T> &other) const noexcept;
    FixedMatrix4<T> operator+(T scalar) const noexcept;
    FixedMatrix4<T> operator-(const FixedMatrix4<T> &other) const noexcept;
    FixedMatrix4<T> operator-(T scalar) const noexcept;
    FixedMatrix4<T> operator*(const FixedMatrix4<T> &other) const noexcept;
    FixedMatrix4<T> operator*(T scalar) const noexcept;
    FixedMatrix4<T> operator/(T scalar) const noexcept;

    FixedMatrix4<T> &operator+=(const FixedMatrix4<T> &other) noexcept;
    FixedMatrix4<T> &operator+=(T scalar) noexcept;
    FixedMatrix4<T> &operator-=(const FixedMatrix4<T> &other) noexcept;
    FixedMatrix4<T> &operator-=(T scalar) noexcept;
    FixedMatrix4<T> &operator*=(const FixedMatrix4<T> &other) noexcept;
    FixedMatrix4<T> &operator*=(T scalar) noexcept;
    FixedMatrix4<T> &operator/=(T scalar) noexcept;
};

template <typename T>
constexpr FixedMatrix4<T>::FixedMatrix4() noexcept {
    for (auto &i : data) {
        i = 0;
    }
}

template <typename T>
template <typename U>
constexpr void FixedMatrix4<T>::construct_with_one(U value) {
    for (auto &i : data) {
        i = value;
    }
}

template <typename T>
template <typename... Ts, typename>
constexpr FixedMatrix4<T>::FixedMatrix4(Ts &&...args) noexcept : data{std::forward<Ts>(args)...} {
    if constexpr (sizeof...(Ts) == 1) {
        construct_with_one(args...);
    } else {
        static_assert(sizeof...(args) == M * N, "Invalid number of arguments");
        T arr[]{args...};
        std::copy(arr, arr + M * N, data);
    };
}

template <typename T>
constexpr FixedMatrix4<T>::FixedMatrix4(const FixedMatrix4<T> &other) noexcept {
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] = other.data[i];
    }
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::operator=(const FixedMatrix4<T> &other) {
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] = other.data[i];
    }
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T>::FixedMatrix4(FixedMatrix4<T> &&other) noexcept {
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] = std::move(other.data[i]);
    }
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::operator=(FixedMatrix4<T> &&other) noexcept {
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] = std::move(other.data[i]);
    }
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::identity() noexcept {
    if constexpr (std::is_integral_v<T>) {
        return FixedMatrix4<T>{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    } else {
        return FixedMatrix4<T>{
            1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    }
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::zero() noexcept {
    if constexpr (std::is_integral_v<T>) {
        FixedMatrix4<T> result{0};
        return result;
    } else {
        FixedMatrix4<T> result{0.0f};
        return result;
    }
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::one() noexcept {
    if constexpr (std::is_integral_v<T>) {
        FixedMatrix4<T> result{1};
        return result;
    } else {
        FixedMatrix4<T> result{1.0f};
        return result;
    }
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::copy() const {
    FixedMatrix4<T> result;
#ifdef MATTY_USE_x86_SIMD_128
    for (std::size_t i = 0; i < 4; i++) {
        value_128bit val = load_128bit(CAST_LOAD_128BIT(data + i * 4));
        store_128bit(CAST_STORE_128BIT(result.data + i * 4), val);
    }
#else
    for (std::size_t i = 0; i < M * N; i++) {
        result.data[i] = data[i];
    }
#endif

    return result;
}

template <typename T>
constexpr T *FixedMatrix4<T>::operator[](std::size_t x) {
    assert(x < M && "Index out of bounds");
    return data + x * N;
}

template <typename T>
constexpr const T *FixedMatrix4<T>::operator[](std::size_t x) const noexcept {
    assert(x < M && "Index out of bounds");
    return data + x * N;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::add(const FixedMatrix4<T> &other) const {
    FixedMatrix4<T> result = this->copy();
    result.add_self(other);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::add_self(const FixedMatrix4<T> &other) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    value_128bit other_row0 = load_128bit(CAST_LOAD_128BIT(other.data));
    value_128bit other_row1 = load_128bit(CAST_LOAD_128BIT(other.data + 4));
    value_128bit other_row2 = load_128bit(CAST_LOAD_128BIT(other.data + 8));
    value_128bit other_row3 = load_128bit(CAST_LOAD_128BIT(other.data + 12));

    store_128bit(CAST_STORE_128BIT(data), add_32bit(row0, other_row0));
    store_128bit(CAST_STORE_128BIT(data + 4), add_32bit(row1, other_row1));
    store_128bit(CAST_STORE_128BIT(data + 8), add_32bit(row2, other_row2));
    store_128bit(CAST_STORE_128BIT(data + 12), add_32bit(row3, other_row3));
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] += other.data[i];
    }
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::add(T scalar) const {
    FixedMatrix4<T> result = this->copy();
    result.add_self(scalar);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::add_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    value_128bit scalar_128bit = set1_32bit(scalar);

    store_128bit(CAST_STORE_128BIT(data), add_32bit(row0, scalar_128bit));
    store_128bit(CAST_STORE_128BIT(data + 4), add_32bit(row1, scalar_128bit));
    store_128bit(CAST_STORE_128BIT(data + 8), add_32bit(row2, scalar_128bit));
    store_128bit(CAST_STORE_128BIT(data + 12), add_32bit(row3, scalar_128bit));
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] += scalar;
    }
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::sub(const FixedMatrix4<T> &other) const {
    FixedMatrix4<T> result = this->copy();
    result.sub_self(other);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::sub_self(const FixedMatrix4<T> &other) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    value_128bit other_row0 = load_128bit(CAST_LOAD_128BIT(other.data));
    value_128bit other_row1 = load_128bit(CAST_LOAD_128BIT(other.data + 4));
    value_128bit other_row2 = load_128bit(CAST_LOAD_128BIT(other.data + 8));
    value_128bit other_row3 = load_128bit(CAST_LOAD_128BIT(other.data + 12));

    store_128bit(CAST_STORE_128BIT(data), sub_32bit(row0, other_row0));
    store_128bit(CAST_STORE_128BIT(data + 4), sub_32bit(row1, other_row1));
    store_128bit(CAST_STORE_128BIT(data + 8), sub_32bit(row2, other_row2));
    store_128bit(CAST_STORE_128BIT(data + 12), sub_32bit(row3, other_row3));
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] -= other.data[i];
    }
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::sub(T scalar) const {
    FixedMatrix4<T> result = this->copy();
    result.sub_self(scalar);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::sub_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    row0 = sub_32bit(row0, set1_32bit(scalar));
    row1 = sub_32bit(row1, set1_32bit(scalar));
    row2 = sub_32bit(row2, set1_32bit(scalar));
    row3 = sub_32bit(row3, set1_32bit(scalar));

    store_128bit(CAST_STORE_128BIT(data), row0);
    store_128bit(CAST_STORE_128BIT(data + 4), row1);
    store_128bit(CAST_STORE_128BIT(data + 8), row2);
    store_128bit(CAST_STORE_128BIT(data + 12), row3);
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] -= scalar;
    }
#endif
    return *this;
}

template <typename T>
template <typename T2>
constexpr FixedMatrix4<T> FixedMatrix4<T>::mul(const FixedMatrix4<T2> &other) const {
    FixedMatrix4<T> result = this->copy();
    result.mul_self(other);
    return result;
}

template <typename T>
template <typename T2>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::mul_self(const FixedMatrix4<T2> &other) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

#define LOAD_COL(i) set_32bit(other.data[i + 12], other.data[i + 8], other.data[i + 4], other.data[i]);
    value_128bit col0 = LOAD_COL(0);
    value_128bit col1 = LOAD_COL(1);
    value_128bit col2 = LOAD_COL(2);
    value_128bit col3 = LOAD_COL(3);
#undef LOAD_COL

#define CALC_ROW_N(n)                                                                                                  \
    do {                                                                                                               \
        result0 = mul_32bit(row##n, col0);                                                                             \
        result1 = mul_32bit(row##n, col1);                                                                             \
        result2 = mul_32bit(row##n, col2);                                                                             \
        result3 = mul_32bit(row##n, col3);                                                                             \
                                                                                                                       \
        result0 = hadd_32bit(result0, result1);                                                                        \
        result1 = hadd_32bit(result2, result3);                                                                        \
        result2 = hadd_32bit(result0, result1);                                                                        \
    } while (0)

    value_128bit result0 = set1_32bit(0);
    value_128bit result1 = set1_32bit(0);
    value_128bit result2 = set1_32bit(0);
    value_128bit result3 = set1_32bit(0);

    CALC_ROW_N(0);
    store_128bit(CAST_STORE_128BIT(data), result2);
    CALC_ROW_N(1);
    store_128bit(CAST_STORE_128BIT(data + 4), result2);
    CALC_ROW_N(2);
    store_128bit(CAST_STORE_128BIT(data + 8), result2);
    CALC_ROW_N(3);
    store_128bit(CAST_STORE_128BIT(data + 12), result2);
#else
    FixedMatrix4<T> temp{0};
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            for (std::size_t k = 0; k < N; k++) {
                temp.data[from_xy_index(i, j)] += other.data[from_xy_index(i, k)] * data[from_xy_index(k, j)];
            }
        }
    }
    *this = temp;
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::mul(T scalar) const {
    FixedMatrix4<T> result = this->copy();
    result.mul_self(scalar);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::mul_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    row0 = mul_32bit(row0, set1_32bit(scalar));
    row1 = mul_32bit(row1, set1_32bit(scalar));
    row2 = mul_32bit(row2, set1_32bit(scalar));
    row3 = mul_32bit(row3, set1_32bit(scalar));

    store_128bit(CAST_STORE_128BIT(data), row0);
    store_128bit(CAST_STORE_128BIT(data + 4), row1);
    store_128bit(CAST_STORE_128BIT(data + 8), row2);
    store_128bit(CAST_STORE_128BIT(data + 12), row3);
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] *= scalar;
    }
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::div(T scalar) const {
    FixedMatrix4<T> result = this->copy();
    result.div_self(scalar);
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::div_self(T scalar) {
    assert(scalar != 0 && "Cannot divide by zero");
#ifdef MATTY_USE_x86_SIMD_128
    if constexpr (std::is_floating_point_v<T>) {
        value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
        value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
        value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
        value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

        row0 = _mm_div_ps(row0, set1_32bit(scalar));
        row1 = _mm_div_ps(row1, set1_32bit(scalar));
        row2 = _mm_div_ps(row2, set1_32bit(scalar));
        row3 = _mm_div_ps(row3, set1_32bit(scalar));

        store_128bit(CAST_STORE_128BIT(data), row0);
        store_128bit(CAST_STORE_128BIT(data + 4), row1);
        store_128bit(CAST_STORE_128BIT(data + 8), row2);
        store_128bit(CAST_STORE_128BIT(data + 12), row3);
    } else {
        for (auto &i : data) {
            i /= scalar;
        }
    }
#else
    for (auto &i : data) {
        i /= scalar;
    }
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::transpose() const {
    FixedMatrix4<T> result = this->copy();
    result.transpose_self();
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::transpose_self() {
#ifdef MATTY_USE_x86_SIMD_128

#define LOAD_COL(i) set_32bit(data[i + 12], data[i + 8], data[i + 4], data[i]);
    value_128bit col0 = LOAD_COL(0);
    value_128bit col1 = LOAD_COL(1);
    value_128bit col2 = LOAD_COL(2);
    value_128bit col3 = LOAD_COL(3);
#undef LOAD_COL

    store_128bit(CAST_STORE_128BIT(data), col0);
    store_128bit(CAST_STORE_128BIT(data + 4), col1);
    store_128bit(CAST_STORE_128BIT(data + 8), col2);
    store_128bit(CAST_STORE_128BIT(data + 12), col3);

#else
    FixedMatrix4<T> temp{0};
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            temp.data[from_xy_index(j, i)] = data[from_xy_index(i, j)];
        }
    }
    *this = temp;
#endif
    return *this;
}

template <typename T>
constexpr std::size_t FixedMatrix4<T>::rows() const noexcept {
    return M;
}

template <typename T>
constexpr std::size_t FixedMatrix4<T>::columns() const noexcept {
    return N;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::horizontal_flip() const {
    FixedMatrix4<T> result = this->copy();
    result.horizontal_flip_self();
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::horizontal_flip_self() {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(CAST_LOAD_128BIT(data));
    value_128bit row1 = load_128bit(CAST_LOAD_128BIT(data + 4));
    value_128bit row2 = load_128bit(CAST_LOAD_128BIT(data + 8));
    value_128bit row3 = load_128bit(CAST_LOAD_128BIT(data + 12));

    store_128bit(CAST_STORE_128BIT(data), shuffle_32bit_op(row0, _MM_SHUFFLE(0, 1, 2, 3)));
    store_128bit(CAST_STORE_128BIT(data + 4), shuffle_32bit_op(row1, _MM_SHUFFLE(0, 1, 2, 3)));
    store_128bit(CAST_STORE_128BIT(data + 8), shuffle_32bit_op(row2, _MM_SHUFFLE(0, 1, 2, 3)));
    store_128bit(CAST_STORE_128BIT(data + 12), shuffle_32bit_op(row3, _MM_SHUFFLE(0, 1, 2, 3)));

#else
    FixedMatrix4<T> temp{0};
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N; j++) {
            temp.data[from_xy_index(i, j)] = data[from_xy_index(i, N - j - 1)];
        }
    }
    *this = temp;
#endif
    return *this;
}

template <typename T>
constexpr FixedMatrix4<T> FixedMatrix4<T>::vertical_flip() const {
    FixedMatrix4<T> result = this->copy();
    result.vertical_flip_self();
    return result;
}

template <typename T>
constexpr FixedMatrix4<T> &FixedMatrix4<T>::vertical_flip_self() {
#ifdef MATTY_USE_x86_SIMD_128
    value_128bit row0 = load_128bit(reinterpret_cast<value_128bit *>(data));
    value_128bit row1 = load_128bit(reinterpret_cast<value_128bit *>(data + 4));
    value_128bit row2 = load_128bit(reinterpret_cast<value_128bit *>(data + 8));
    value_128bit row3 = load_128bit(reinterpret_cast<value_128bit *>(data + 12));

    store_128bit(CAST_STORE_128BIT(data + 12), row0);
    store_128bit(CAST_STORE_128BIT(data + 8), row1);
    store_128bit(CAST_STORE_128BIT(data + 4), row2);
    store_128bit(CAST_STORE_128BIT(data), row3);
#else
    for (std::size_t i = 0; i < M / 2; i++) {
        for (std::size_t j = 0; j < N; j++) {
            std::swap(data[from_xy_index(i, j)], data[from_xy_index(M - i - 1, j)]);
        }
    }
#endif
    return *this;
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator+(const FixedMatrix4<T> &other) const noexcept {
    return this->copy().add(other);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator+(T scalar) const noexcept {
    return this->copy().add(scalar);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator-(const FixedMatrix4<T> &other) const noexcept {
    return this->copy().sub(other);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator-(T scalar) const noexcept {
    return this->copy().sub(scalar);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator*(const FixedMatrix4<T> &other) const noexcept {
    return this->copy().mul(other);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator*(T scalar) const noexcept {
    return this->copy().mul(scalar);
}

template <typename T>
FixedMatrix4<T> FixedMatrix4<T>::operator/(T scalar) const noexcept {
    return this->copy().div(scalar);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator+=(const FixedMatrix4<T> &other) noexcept {
    return add_self(other);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator+=(T scalar) noexcept {
    return add_self(scalar);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator-=(const FixedMatrix4<T> &other) noexcept {
    return sub_self(other);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator-=(T scalar) noexcept {
    return sub_self(scalar);
}
template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator*=(const FixedMatrix4<T> &other) noexcept {
    return mul_self(other);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator*=(T scalar) noexcept {
    return mul_self(scalar);
}

template <typename T>
FixedMatrix4<T> &FixedMatrix4<T>::operator/=(T scalar) noexcept {
    return div_self(scalar);
}
} // namespace matty

#undef DBGPRINT

#endif
