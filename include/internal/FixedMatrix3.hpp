#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef MATTY_FIXED_MATRIX_3_HPP
#define MATTY_FIXED_MATRIX_3_HPP

#include "includes.hpp"

namespace matty {
template <typename T>
class FixedMatrix3 {
    constexpr static std::size_t M = 4;
    constexpr static std::size_t N = 4;

    T data[M * N];

    constexpr std::size_t from_xy_index(std::size_t x, std::size_t y) { return x * 4 + y; }

    static_assert(CHAR_BIT == 8 && "Only 8-bit bytes supported");
    static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Only 32-bit or 64-bit types supported");

    template <typename U>
    constexpr void construct_with_one(U value);

  public:
    constexpr FixedMatrix3() noexcept;

    template <typename... Ts, typename = std::enable_if_t<(std::is_convertible_v<Ts, T> && ...)>>
    constexpr explicit FixedMatrix3(Ts &&...args) noexcept;

    constexpr FixedMatrix3(const FixedMatrix3<T> &other) noexcept;
    constexpr FixedMatrix3<T> &operator=(const FixedMatrix3<T> &other);

    constexpr FixedMatrix3(FixedMatrix3<T> &&other) noexcept;
    constexpr FixedMatrix3<T> &operator=(FixedMatrix3<T> &&other) noexcept;

    constexpr static FixedMatrix3<T> identity() noexcept;
    constexpr static FixedMatrix3<T> zero() noexcept;
    constexpr static FixedMatrix3<T> one() noexcept;

    constexpr FixedMatrix3<T> copy() const;
    constexpr T *operator[](std::size_t x);
    constexpr const T *operator[](std::size_t x) const noexcept;

    constexpr FixedMatrix3<T> add(const FixedMatrix3<T> &other) const;
    constexpr void add_self(const FixedMatrix3<T> &other);

    constexpr FixedMatrix3<T> sub(const FixedMatrix3<T> &other) const;
    constexpr void sub_self(const FixedMatrix3<T> &other);

    template <typename T2 = int>
    constexpr FixedMatrix3<T> mul(const FixedMatrix3<T2> &other) const;
    template <typename T2 = int>
    constexpr void mul_self(const FixedMatrix3<T2> &other);

    constexpr FixedMatrix3<T> transpose() const;
    constexpr void transpose_self();

    constexpr FixedMatrix3<T> horizontal_flip() const;
    constexpr void horizontal_flip_self();

    constexpr FixedMatrix3<T> vertical_flip() const;
    constexpr void vertical_flip_self();

    [[nodiscard]] constexpr std::size_t rows() const noexcept;
    [[nodiscard]] constexpr std::size_t columns() const noexcept;
};
} // namespace matty

#endif