#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef MATTY_DYNAMIC_MATRIX_HPP
#define MATTY_DYNAMIC_MATRIX_HPP

#include "includes.hpp"
#include "internal/Aliases.hpp"

namespace matty {
template <typename T = int>
class DynamicMatrix : Aliases<T> {
    USING_ALL_ALIASES

#ifdef MATTY_USE_x86_SIMD_128
    static constexpr std::size_t elements_per_vector = sizeof(value_128bit) / sizeof(T);
#endif

    std::unique_ptr<T[]> data{};
    std::size_t M{};
    std::size_t N{};

    [[nodiscard]] std::size_t from_xy_index(std::size_t x, std::size_t y) const noexcept;

  public:
    DynamicMatrix(std::size_t M, std::size_t N);

    template <typename... Ts, typename = std::enable_if_t<(std::is_integral_v<Ts> && ...)>>
    explicit DynamicMatrix(std::pair<std::size_t, std::size_t> dimensions, Ts &&...args);

    DynamicMatrix(const DynamicMatrix<T> &other) noexcept;
    DynamicMatrix<T> &operator=(const DynamicMatrix<T> &other);

    DynamicMatrix(DynamicMatrix<T> &&other) noexcept;
    DynamicMatrix<T> &operator=(DynamicMatrix<T> &&other) noexcept;

    static DynamicMatrix<T> identity(std::size_t M, std::size_t N) noexcept;
    static DynamicMatrix<T> zero(std::size_t M, std::size_t N) noexcept;
    static DynamicMatrix<T> one(std::size_t M, std::size_t N) noexcept;

    DynamicMatrix<T> copy() const;
    T *operator[](std::size_t x);
    const T *operator[](std::size_t x) const noexcept;

    DynamicMatrix<T> add(const DynamicMatrix<T> &other) const;
    DynamicMatrix<T> &add_self(const DynamicMatrix<T> &other);

    DynamicMatrix<T> add(T scalar) const;
    DynamicMatrix<T> &add_self(T scalar);

    DynamicMatrix<T> sub(const DynamicMatrix<T> &other) const;
    DynamicMatrix<T> &sub_self(const DynamicMatrix<T> &other);

    DynamicMatrix<T> sub(T scalar) const;
    DynamicMatrix<T> &sub_self(T scalar);

    template <typename T2 = int>
    DynamicMatrix<T> mul(const DynamicMatrix<T2> &other) const;

    DynamicMatrix<T> mul(T scalar) const;
    DynamicMatrix<T> &mul_self(T scalar);

    DynamicMatrix<T> div(T scalar) const;
    DynamicMatrix<T> &div_self(T scalar);

    DynamicMatrix<T> transpose() const;
    void transpose_self();

    DynamicMatrix<T> horizontal_flip() const;
    void horizontal_flip_self();

    DynamicMatrix<T> vertical_flip() const;
    void vertical_flip_self();

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t columns() const noexcept;

    DynamicMatrix<T> operator+(const DynamicMatrix<T> &other) const noexcept;
    DynamicMatrix<T> operator+(T scalar) const noexcept;
    DynamicMatrix<T> operator-(const DynamicMatrix<T> &other) const noexcept;
    DynamicMatrix<T> operator-(T scalar) const noexcept;
    DynamicMatrix<T> operator*(const DynamicMatrix<T> &other) const noexcept;
    DynamicMatrix<T> operator*(T scalar) const noexcept;
    DynamicMatrix<T> operator/(T scalar) const noexcept;

    DynamicMatrix<T> &operator+=(const DynamicMatrix<T> &other) noexcept;
    DynamicMatrix<T> &operator+=(T scalar) noexcept;
    DynamicMatrix<T> &operator-=(const DynamicMatrix<T> &other) noexcept;
    DynamicMatrix<T> &operator-=(T scalar) noexcept;
    DynamicMatrix<T> &operator*=(const DynamicMatrix<T> &other) noexcept;
    DynamicMatrix<T> &operator*=(T scalar) noexcept;
    DynamicMatrix<T> &operator/=(T scalar) noexcept;

    bool check_dimensions(const DynamicMatrix<T> &other) const noexcept;
};

template <typename T>
DynamicMatrix<T>::DynamicMatrix(std::size_t M, std::size_t N) {
    assert(M > 0 && "Matrix must have number of rows > 0");
    assert(N > 0 && "Matrix must have number of columns > 0");

    this->M = M;
    this->N = N;

#ifdef MATTY_USE_x86_SIMD_128
    T *memory = new (std::align_val_t(std::max(alignof(T), alignof(__m128i)))) T[M * N];
    std::uninitialized_fill_n(memory, M * N, T{});
    data.reset(memory);
#else
    data = std::make_unique<T[]>(M * N);
#endif
}

template <typename T>
template <typename... Ts, typename>
DynamicMatrix<T>::DynamicMatrix(std::pair<std::size_t, std::size_t> dimensions, Ts &&...args) {
    assert(dimensions.first > 0 && "Matrix must have number of rows > 0");
    assert(dimensions.second > 0 && "Matrix must have number of columns > 0");

    this->M = dimensions.first;
    this->N = dimensions.second;

#ifdef MATTY_USE_x86_SIMD_128
    T *memory = new (std::align_val_t(std::max(alignof(T), alignof(__m128i))))
        T[dimensions.first * dimensions.second]{std::forward<Ts>(args)...};
    data.reset(memory);
#else
    data = std::make_unique<T[]>(dimensions.first * dimensions.second);
    std::size_t i = 0;
    ((data[i++] = std::forward<Ts>(args)), ...);
#endif
}

template <typename T>
DynamicMatrix<T>::DynamicMatrix(const DynamicMatrix<T> &other) noexcept {
    this->M = other.M;
    this->N = other.N;

#ifdef MATTY_USE_x86_SIMD_128
    T *memory = new (std::align_val_t(std::max(alignof(T), alignof(__m128i)))) T[M * N];
    std::size_t i = 0;
    for (; i < M * N - sizeof(value_128bit) + 1; i += sizeof(value_128bit)) {
        value_128bit val = load_128bit(CAST_LOAD_128BIT(other.data.get() + i));
        store_128bit(CAST_STORE_128BIT(memory + i), val);
    }
    for (; i < M * N; i++) {
        memory[i] = other.data[i];
    }
    data.reset(memory);
#else
    data = std::make_unique<T[]>(other.M * other.N);
    std::copy(other.data.get(), other.data.get() + other.M * other.N, data.get());
#endif
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator=(const DynamicMatrix<T> &other) {
    this->M = other.M;
    this->N = other.N;

#ifdef MATTY_USE_x86_SIMD_128
    T *memory = new (std::align_val_t(std::max(alignof(T), alignof(__m128i)))) T[M * N];
    std::size_t i = 0;
    for (; i < M * N - sizeof(value_128bit) + 1; i += sizeof(value_128bit)) {
        value_128bit val = load_128bit(CAST_LOAD_128BIT(other.data.get() + i));
        store_128bit(CAST_STORE_128BIT(memory + i), val);
    }
    for (; i < M * N; i++) {
        memory[i] = other.data[i];
    }
    data.reset(memory);
#else
    data.reset();
    data = std::make_unique<T[]>(other.M * other.N);
    std::copy(other.data.get(), other.data.get() + other.M * other.N, data.get());
#endif

    return *this;
}

template <typename T>
DynamicMatrix<T>::DynamicMatrix(DynamicMatrix<T> &&other) noexcept {
    this->M = other.M;
    this->N = other.N;
    data = std::move(other.data);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator=(DynamicMatrix<T> &&other) noexcept {
    this->M = other.M;
    this->N = other.N;
    data = std::move(other.data);
    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::identity(std::size_t M, std::size_t N) noexcept {
    DynamicMatrix<T> result(M, N);
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    for (; i < M * N - sizeof(value_128bit) + 1; i += sizeof(value_128bit)) {
        store_128bit(CAST_STORE_128BIT(result.data.get() + i), set1_32bit(0));
    }
    for (; i < M * N; i++) {
        result.data[i] = T{0};
    }
    for (std::size_t j = 0; j < std::min(M, N); j++) {
        result[j][j] = T{1};
    }
#else
    std::uninitialized_fill_n(result.data.get(), M * N, T{0});
    for (std::size_t i = 0; i < std::min(M, N); i++) {
        result[i][i] = T{1};
    }
#endif
    return result;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::zero(std::size_t M, std::size_t N) noexcept {
    DynamicMatrix<T> result(M, N);
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    for (; i < M * N - sizeof(value_128bit) + 1; i += sizeof(value_128bit)) {
        store_128bit(CAST_STORE_128BIT(result.data.get() + i), set1_32bit(0));
    }
    for (; i < M * N; i++) {
        result.data[i] = T{0};
    }
#else
    std::uninitialized_fill_n(result.data.get(), M * N, T{0});
#endif
    return result;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::one(std::size_t M, std::size_t N) noexcept {
    DynamicMatrix<T> result(M, N);
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    for (; i < M * N - sizeof(value_128bit) + 1; i += sizeof(value_128bit)) {
        store_128bit(CAST_STORE_128BIT(result.data.get() + i), set1_32bit(1));
    }
    for (; i < M * N; i++) {
        result.data[i] = T{1};
    }
#else
    std::uninitialized_fill_n(result.data.get(), M * N, T{1});
#endif
    return result;
}

template <typename T>
std::size_t DynamicMatrix<T>::from_xy_index(std::size_t x, std::size_t y) const noexcept {
    return N * x + y;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::copy() const {
    DynamicMatrix<T> copy{M, N};
    std::copy(data.get(), data.get() + M * N, copy.data.get());
    return copy;
}

template <typename T>
T *DynamicMatrix<T>::operator[](std::size_t x) {
    return data.get() + from_xy_index(x, 0);
}

template <typename T>
const T *DynamicMatrix<T>::operator[](std::size_t x) const noexcept {
    return data.get() + from_xy_index(x, 0);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::add(const DynamicMatrix<T> &other) const {
    assert(check_dimensions(other) && "Matrices must have same dimensions");
    DynamicMatrix<T> result = this->copy();
    result.add_self(other);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::add_self(const DynamicMatrix<T> &other) {
    assert(check_dimensions(other) && "Matrices must have same dimensions");

    std::size_t i = 0;

#ifdef MATTY_USE_x86_SIMD_128
    std::size_t remainder = (M * N) % elements_per_vector;
    for (; i < (M * N) - remainder; i += elements_per_vector) {
        value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
        value_128bit second = load_128bit(CAST_LOAD_128BIT(other.data.get() + i));
        value_128bit added = add_32bit(first, second);
        store_128bit(CAST_STORE_128BIT(data.get() + i), added);
    }
#endif

    for (; i < M * N; i++) {
        data[i] += other.data[i];
    }

    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::add(T scalar) const {
    DynamicMatrix<T> result = this->copy();
    result.add_self(scalar);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::add_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    std::size_t remainder = (M * N) % elements_per_vector;
    for (; i < (M * N) - remainder; i += elements_per_vector) {
        value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
        value_128bit added = add_32bit(first, set1_32bit(scalar));
        store_128bit(CAST_STORE_128BIT(data.get() + i), added);
    }
    for (; i < M * N; i++) {
        data[i] += scalar;
    }
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] += scalar;
    }
#endif
    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::sub(const DynamicMatrix<T> &other) const {
    assert(check_dimensions(other) && "Matrices must have same dimensions");
    DynamicMatrix<T> result = this->copy();
    result.sub_self(other);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::sub_self(const DynamicMatrix<T> &other) {
    assert(check_dimensions(other) && "Matrices must have same dimensions");

    std::size_t i = 0;

#ifdef MATTY_USE_x86_SIMD_128
    constexpr std::size_t remainder = (M * N) % elements_per_vector;
    for (; i < (M * N) - remainder; i += elements_per_vector) {
        value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
        value_128bit second = load_128bit(CAST_LOAD_128BIT(other.data.get() + i));
        value_128bit subtracted = sub_32bit(first, second);
        store_128bit(CAST_STORE_128BIT(data.get() + i), subtracted);
    }
#endif

    for (; i < M * N; i++) {
        data[i] += other.data[i];
    }

    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::sub(T scalar) const {
    DynamicMatrix<T> result = this->copy();
    result.sub_self(scalar);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::sub_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    std::size_t remainder = (M * N) % elements_per_vector;
    for (; i < (M * N) - remainder; i += elements_per_vector) {
        value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
        value_128bit subtracted = sub_32bit(first, set1_32bit(scalar));
        store_128bit(CAST_STORE_128BIT(data.get() + i), subtracted);
    }
    for (; i < M * N; i++) {
        data[i] -= scalar;
    }
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] -= scalar;
    }
#endif
    return *this;
}

template <typename T>
template <typename T2>
DynamicMatrix<T> DynamicMatrix<T>::mul(const DynamicMatrix<T2> &other) const {
    DynamicMatrix<T> result{M, other.N};

    assert(N == other.M && "Matrices must have compatible dimensions");

#ifdef MATTY_USE_x86_SIMD_128
    std::size_t remainder = other.N % elements_per_vector;
    assert((other.N - remainder) % elements_per_vector == 0 &&
           "(other.N - remainder) must be a multiple of elements_per_vector");

    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t k = 0; k < N; k++) {
            for (std::size_t j = 0; j < other.N - remainder; j += elements_per_vector) {
                value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + from_xy_index(i, k)));
                value_128bit second = load_128bit(CAST_LOAD_128BIT(other.data.get() + from_xy_index(k, j)));
                value_128bit multiplied = mul_32bit(first, second);
                store_128bit(CAST_STORE_128BIT(result.data.get() + from_xy_index(i, j)), multiplied);
            }
        }
    }

    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t k = 0; k < N; k++) {
            for (std::size_t j = other.N - remainder; j < other.N; j++) {
                result[i][j] += (*this)[i][k] * other[k][j];
            }
        }
    }
#else
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t k = 0; k < N; k++) {
            for (std::size_t j = 0; j < other.N; j++) {
                result[i][j] += (*this)[i][k] * other[k][j];
            }
        }
    }
#endif

    return result;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::mul(T scalar) const {
    DynamicMatrix<T> result = this->copy();
    result.mul_self(scalar);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::mul_self(T scalar) {
#ifdef MATTY_USE_x86_SIMD_128
    std::size_t i = 0;
    std::size_t remainder = (M * N) % elements_per_vector;
    for (; i < (M * N) - remainder; i += elements_per_vector) {
        value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
        value_128bit multiplied = mul_32bit(first, set1_32bit(scalar));
        store_128bit(CAST_STORE_128BIT(data.get() + i), multiplied);
    }
    for (; i < M * N; i++) {
        data[i] *= scalar;
    }
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] *= scalar;
    }
#endif
    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::div(T scalar) const {
    DynamicMatrix<T> result = this->copy();
    result.div_self(scalar);
    return result;
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::div_self(T scalar) {
    assert(scalar != 0 && "Division by zero");
#ifdef MATTY_USE_x86_SIMD_128
    if constexpr (std::is_floating_point_v<T>) {
        std::size_t i = 0;
        std::size_t remainder = (M * N) % elements_per_vector;
        for (; i < (M * N) - remainder; i += elements_per_vector) {
            value_128bit first = load_128bit(CAST_LOAD_128BIT(data.get() + i));
            value_128bit divided = _mm_div_ps(first, set1_32bit(scalar));
            store_128bit(CAST_STORE_128BIT(data.get() + i), divided);
        }
        for (; i < M * N; i++) {
            data[i] /= scalar;
        }
    } else {
        for (std::size_t i = 0; i < M * N; i++) {
            data[i] /= scalar;
        }
    }
#else
    for (std::size_t i = 0; i < M * N; i++) {
        data[i] /= scalar;
    }
#endif
    return *this;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::transpose() const {
    assert(M == N && "Matrix must be square");
    DynamicMatrix<T> result = this->copy();
    result.transpose_self();
    return result;
}

template <typename T>
void DynamicMatrix<T>::transpose_self() {
    assert(M == N && "Matrix must be square");

    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = i + 1; j < N; j++) {
            std::swap(data[from_xy_index(i, j)], data[from_xy_index(j, i)]);
        }
    }
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::horizontal_flip() const {
    DynamicMatrix<T> result = this->copy();
    result.horizontal_flip_self();
    return result;
}

template <typename T>
void DynamicMatrix<T>::horizontal_flip_self() {
    for (std::size_t i = 0; i < M; i++) {
        for (std::size_t j = 0; j < N / 2; j++) {
            std::swap(data[from_xy_index(i, j)], data[from_xy_index(i, N - j - 1)]);
        }
    }
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::vertical_flip() const {
    DynamicMatrix<T> result = this->copy();
    result.vertical_flip_self();
    return result;
}

template <typename T>
void DynamicMatrix<T>::vertical_flip_self() {
    for (std::size_t i = 0; i < M / 2; i++) {
        for (std::size_t j = 0; j < N; j++) {
            std::swap(data[from_xy_index(i, j)], data[from_xy_index(M - i - 1, j)]);
        }
    }
}

template <typename T>
std::size_t DynamicMatrix<T>::rows() const noexcept {
    return M;
}

template <typename T>
std::size_t DynamicMatrix<T>::columns() const noexcept {
    return N;
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator+(const DynamicMatrix<T> &other) const noexcept {
    return this->add(other);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator+(T scalar) const noexcept {
    return this->add(scalar);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator-(const DynamicMatrix<T> &other) const noexcept {
    return this->sub(other);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator-(T scalar) const noexcept {
    return this->sub(scalar);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator*(const DynamicMatrix<T> &other) const noexcept {
    return this->mul(other);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator*(T scalar) const noexcept {
    return this->mul(scalar);
}

template <typename T>
DynamicMatrix<T> DynamicMatrix<T>::operator/(T scalar) const noexcept {
    return this->div(scalar);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator+=(const DynamicMatrix<T> &other) noexcept {
    return this->add_self(other);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator+=(T scalar) noexcept {
    return this->add_self(scalar);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator-=(const DynamicMatrix<T> &other) noexcept {
    return this->sub_self(other);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator-=(T scalar) noexcept {
    return this->sub_self(scalar);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator*=(const DynamicMatrix<T> &other) noexcept {
    return this->mul_self(other);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator*=(T scalar) noexcept {
    return this->mul_self(scalar);
}

template <typename T>
DynamicMatrix<T> &DynamicMatrix<T>::operator/=(T scalar) noexcept {
    return this->div_self(scalar);
}

template <typename T>
bool DynamicMatrix<T>::check_dimensions(const DynamicMatrix<T> &other) const noexcept {
    return M == other.M && N == other.N;
}

} // namespace matty

#endif