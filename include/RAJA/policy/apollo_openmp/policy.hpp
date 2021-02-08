/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA apollo openmp policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#ifndef policy_apollo_openmp_HPP
#define policy_apollo_openmp_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace policy
{
namespace apollo_omp
{

//
//////////////////////////////////////////////////////////////////////
//
// Execution policies
//
//////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

struct apollo_omp_parallel_for_exec : make_policy_pattern_launch_platform_t<Policy::apollo_openmp,
                                                        Pattern::forall,
                                                        Launch::undefined,
                                                        Platform::host> {
};

///
/// Index set segment iteration policies
///
using apollo_omp_segit = apollo_omp_parallel_for_exec;

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///
struct apollo_omp_reduce
    : make_policy_pattern_t<Policy::apollo_openmp, Pattern::reduce> {
};

const int POLICY_COUNT = 20;

}  // end namespace apollo_omp
}  // end namespace policy

using policy::apollo_omp::apollo_omp_parallel_for_exec;
using policy::apollo_omp::apollo_omp_segit;
using policy::apollo_omp::apollo_omp_reduce;

}  // end namespace RAJA

#endif
