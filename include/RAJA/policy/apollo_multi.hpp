/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for apollo_multi execution.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_apollo_multi_HPP
#define RAJA_apollo_multi_HPP

#if defined(RAJA_ENABLE_APOLLO)

#include "apollo/Apollo.h"
#include "apollo/Region.h"

#if defined(RAJA_CUDA_ACTIVE)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if defined(RAJA_HIP_ACTIVE)
#include <hip/hip_runtime.h>
#endif

#include "RAJA/policy/apollo_multi/forall.hpp"
#include "RAJA/policy/apollo_multi/kernel.hpp"
//#include "RAJA/policy/apollo_multi/policy.hpp"
#include "RAJA/policy/apollo_multi/scan.hpp"

#endif // RAJA_ENABLE_APOLLO

#endif  // closing endif for header file include guard
