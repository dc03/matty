#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef MATTY_FIXED_MATRIX_HPP
#define MATTY_FIXED_MATRIX_HPP

#include "includes.hpp"

namespace matty {
template <typename T>
class FixedMatrix3;
template <typename T>
class FixedMatrix4;
} // namespace matty

#include "internal/FixedMatrix3.hpp"
#include "internal/FixedMatrix4.hpp"

#endif
