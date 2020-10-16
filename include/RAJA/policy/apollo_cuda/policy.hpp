/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA CUDA policy definitions.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_apollo_cuda_HPP
#define RAJA_policy_apollo_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE) && defined(RAJA_ENABLE_APOLLO)

#include <utility>

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace policy
{
namespace cuda
{

template <size_t BLOCK_SIZE, bool Async = false>
struct apollo_cuda_exec : public RAJA::make_policy_pattern_launch_platform_t<
                       RAJA::Policy::cuda,
                       RAJA::Pattern::forall,
                       detail::get_launch<Async>::value,
                       RAJA::Platform::cuda> {
};


}  // end namespace apollo_cuda
}  // end namespace policy

using policy::cuda::apollo_cuda_exec;

template <size_t BLOCK_SIZE>
using apollo_cuda_exec_async = policy::cuda::apollo_cuda_exec<BLOCK_SIZE, true>;

}  // namespace RAJA

#endif  // RAJA_ENABLE_CUDA && RAJA_ENABLE_APOLLO
#endif
