//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//
// Fundamental includes and structs used throughout RAJA tests.
//

#ifndef __RAJA_test_base_HPP__
#define __RAJA_test_base_HPP__

#include "RAJA/RAJA.hpp"

#include "gtest/gtest.h"

//
// Unroll types for gtest testing::Types
//
template <class T>
struct Test;

template <class... T>
struct Test<camp::list<T...>> {
  using Types = ::testing::Types<T...>;
};


#endif // __RAJA_test_base_HPP__
