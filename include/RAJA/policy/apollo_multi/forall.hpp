/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index set and segment iteration
 *          template methods for Apollo-guided execution.
 *
 *          These methods should work on any platform.
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

#ifndef RAJA_forall_apollo_multi_HPP
#define RAJA_forall_apollo_multi_HPP

#include <omp.h>

#include <functional>
#include <type_traits>

#include <string>
#include <sstream>
#include <functional>
#include <unordered_set>

#include <omp.h>

#include "RAJA/util/types.hpp"

#include "RAJA/index/IndexSet.hpp"
#include "RAJA/index/ListSegment.hpp"
#include "RAJA/index/RangeSegment.hpp"

#include "RAJA/pattern/forall.hpp"
#include "RAJA/pattern/region.hpp"

#include "RAJA/policy/apollo_multi/policy.hpp"
#include "RAJA/internal/fault_tolerance.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"


// ----------


namespace RAJA
{
namespace policy
{
namespace apollo_multi
{
//
//////////////////////////////////////////////////////////////////////
//
// The following function template switches between various RAJA
// execution policies based on feedback from the Apollo system.
//
//////////////////////////////////////////////////////////////////////
//

#include <iostream> // ggout

template <typename Pol>
static RAJA_INLINE
concepts::enable_if<
  type_traits::is_cuda_policy<Pol>,
  RAJA::launch_is<Pol, RAJA::Launch::async>
>
PreLaunch(resources::Cuda &cuda_res, Apollo::Region *region, Apollo::RegionContext *context) {
  using callback_t = RAJA::apollo_cuda::ApolloCallbackDataPool::callback_t;
  callback_t *cbdata =
      reinterpret_cast<callback_t *>(region->callback_pool->get());
  context->isDoneCallback = RAJA::apollo_cuda::ApolloCallbackHelper::isDoneCallback;
  context->callback_arg = cbdata;
  //std::cout << "prelaunch cuda async -> setting event_record and context\n";
  cudaEventRecord(cbdata->start, cuda_res.get_stream());
}

template <typename Pol>
static RAJA_INLINE
concepts::enable_if<
  type_traits::is_cuda_policy<Pol>,
  RAJA::launch_is<Pol, RAJA::Launch::async>
>
PostLaunch(resources::Cuda &cuda_res, Apollo::Region *region, Apollo::RegionContext *context) {
  using callback_t = RAJA::apollo_cuda::ApolloCallbackDataPool::callback_t;
  callback_t *cbdata = reinterpret_cast<callback_t *>(context->callback_arg);
  cudaEventRecord(cbdata->stop, cuda_res.get_stream());
  region->end(context);
  //std::cout << "postlaunch cuda async -> call region->end() with callback and cbdata\n";
}

template <typename Pol, typename Res>
static RAJA_INLINE void PreLaunch(Res &r, Apollo::Region *region, Apollo::RegionContext *context) {
  //std::cout << "prelaunch default -> noop\n";
  // default noop
}

template <typename Pol, typename Res>
static RAJA_INLINE void PostLaunch(Res &r, Apollo::Region *region, Apollo::RegionContext *context) {
  //std::cout << "postlaunch default -> calling region->end(); \n";
  region->end(context);
}

template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename PolicyList,
          typename Iterable,
          typename Func>
struct PolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iterable &&iter,
                                   Func &&loop_body)
  {
    using ExecutionPolicy = camp::at_v<PolicyList, idx>;

    if (policy == idx) {
      // generate policy variant, calls top-level forall pattern.
      auto r = resources::get_resource<ExecutionPolicy>::type::get_default();

      PreLaunch<ExecutionPolicy>(r, region, context);

      RAJA::forall<ExecutionPolicy>(r,
                                    std::forward<Iterable>(iter),
                                    std::move(loop_body));

      PostLaunch<ExecutionPolicy>(r, region, context);
    } else
      PolicyGeneratorSingle<idx + 1, num_policies, PolicyList, Iterable, Func>::
          generate(policy, region, context, std::forward<Iterable>(iter), std::move(loop_body));
  }
};

template <camp::idx_t num_policies,
          typename PolicyList,
          typename Iterable,
          typename Func>
struct PolicyGeneratorSingle<num_policies,
                             num_policies,
                             PolicyList,
                             Iterable,
                             Func> {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iterable &&iter,
                                   Func &&loop_body)
  {
  }
};

template <typename PolicyList, typename Iterable, typename Func>
static RAJA_INLINE void PolicyGenerator(int policy,
                                        Apollo::Region *region,
                                        Apollo::RegionContext *context,
                                        Iterable &&iter,
                                        Func &&loop_body)
{
  PolicyGeneratorSingle<0,
                        camp::size<PolicyList>::value,
                        PolicyList,
                        Iterable,
                        Func>::generate(policy,
                                        region,
                                        context,
                                        std::forward<Iterable>(iter),
                                        std::move(loop_body));
}

// TODO: fix resources, should not be host, may be cuda etc.
template <typename PolicyList, typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const apollo_multi_exec<PolicyList> &,
                             Iterable &&iter,
                             Func &&loop_body)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  // TODO: Move the callback data pool to generic apollo header
  static RAJA::apollo_cuda::ApolloCallbackDataPool *callback_pool = nullptr;
  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    callback_pool = new RAJA::apollo_cuda::ApolloCallbackDataPool(64, 64);
    apolloRegion =
        new Apollo::Region(/* num features */ 1,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<PolicyList>::value,
                           /* callback pool */ callback_pool
        );
  }

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(std::begin(iter), std::end(iter));

  Apollo::RegionContext *context = apolloRegion->begin({num_elements});

  policy_index = apolloRegion->getPolicyIndex(context);
  //std::cout << "policy_index " << policy_index << std::endl;  // ggout

  PolicyGenerator<PolicyList, Iterable, Func>(policy_index,
                                              apolloRegion,
                                              context,
                                              std::forward<Iterable>(iter),
                                              std::move(loop_body));

  return;  // resources::EventProxy<Res>(&res);
}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
