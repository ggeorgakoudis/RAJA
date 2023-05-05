/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA apollo_multi policy definitions.
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

#ifndef policy_apollo_multi_HPP
#define policy_apollo_multi_HPP

#include "RAJA/policy/PolicyBase.hpp"

namespace RAJA
{
namespace policy
{
namespace apollo_multi
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

template <typename...Policies>
using PolicyList = camp::list<Policies...>;

template <typename PolicyList>
struct apollo_multi_exec : public RAJA::make_policy_pattern_launch_platform_t<
                                RAJA::Policy::apollo_multi,
                                RAJA::Pattern::forall,
                                Launch::undefined,
                                RAJA::Platform::undefined> {
};

template <size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async = false>
struct hip_exec_apollo : public RAJA::make_policy_pattern_launch_platform_t<
                            RAJA::Policy::hip,
                            RAJA::Pattern::forall,
                            detail::get_launch<Async>::value,
                            RAJA::Platform::hip> {
};

template <size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async = false>
struct hip_exec_apollo_runtime
    : public RAJA::make_policy_pattern_launch_platform_t<
          RAJA::Policy::hip,
          RAJA::Pattern::forall,
          detail::get_launch<Async>::value,
          RAJA::Platform::hip> {
};

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP>
struct hip_launch_apollo_range_t
    : public RAJA::make_policy_pattern_launch_platform_t<
          RAJA::Policy::hip,
          RAJA::Pattern::region,
          detail::get_launch<Async>::value,
          RAJA::Platform::hip> {
};

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction execution policies
///
///////////////////////////////////////////////////////////////////////
///

}  // end namespace apollo_multi
}  // end namespace policy

using policy::apollo_multi::apollo_multi_exec;
using policy::apollo_multi::hip_exec_apollo;
using policy::apollo_multi::hip_exec_apollo_runtime;
using policy::apollo_multi::hip_launch_apollo_range_t;

namespace resources
{
template <size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async>
struct get_resource<
    hip_exec_apollo<BLOCK_SIZE_START, BLOCK_SIZE_END, BLOCK_SIZE_STEP, Async>> {

  using type = camp::resources::Hip;
};

template <size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async>
struct get_resource<hip_exec_apollo_runtime<BLOCK_SIZE_START,
                                            BLOCK_SIZE_END,
                                            BLOCK_SIZE_STEP,
                                            Async>> {
  using type = camp::resources::Hip;
};

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP>
struct get_resource<hip_launch_apollo_range_t<Async,
                                              GRID_SIZE_START,
                                              GRID_SIZE_END,
                                              GRID_SIZE_STEP,
                                              BLOCK_SIZE_START,
                                              BLOCK_SIZE_END,
                                              BLOCK_SIZE_STEP>> {
  using type = camp::resources::Hip;
};


} // end namespace resources

///
///////////////////////////////////////////////////////////////////////
///
/// Shared memory policies
///
///////////////////////////////////////////////////////////////////////
///

}  // closing brace for RAJA namespace

#endif
