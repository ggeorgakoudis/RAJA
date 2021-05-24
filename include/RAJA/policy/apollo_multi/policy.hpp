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

///
///////////////////////////////////////////////////////////////////////
///
/// Shared memory policies
///
///////////////////////////////////////////////////////////////////////
///

}  // closing brace for RAJA namespace

#endif
