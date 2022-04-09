#pragma once

/* Copyright (C) 2020-2022  Dhruv Chawla */
/* See LICENSE at project root for license details */

#ifndef MATTY_INCLUDES_HPP
#define MATTY_INCLUDES_HPP

#include <cassert>
#include <climits>
#include <cstddef>
#include <initializer_list>
#include <memory>

#define MATTY_USE_x86_SIMD_128

#ifdef MATTY_USE_x86_SIMD_128
#define MATTY_SIMD_STATUS      "128-bit SIMD enabled"
#define MATTY_SIMD_WIDTH_BITS  128
#define MATTY_SIMD_WIDTH_BYTES 16
#include <cstdlib>
#include <type_traits>
#include <x86intrin.h>
#endif

#endif