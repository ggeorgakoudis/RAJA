/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing Apollo wrapper to RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_scan_apollo_HPP
#define RAJA_scan_apollo_HPP

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include "RAJA/config.hpp"
#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA
{
namespace impl
{
namespace scan
{


/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/

#if defined(RAJA_ENABLE_CUDA)
template <typename Pol>
static RAJA_INLINE concepts::enable_if<
    type_traits::is_cuda_policy<Pol>,
    RAJA::launch_is<Pol, RAJA::Launch::async> >
PreLaunch(resources::Cuda & /*cuda_res*/,
          Apollo::Region * /*region*/,
          Apollo::RegionContext *context)
{
  context->timer = Apollo::Timer::create<Apollo::Timer::CudaAsync>();
  context->timer->start();
}
#endif

#if defined(RAJA_ENABLE_HIP)
template <typename Pol>
static RAJA_INLINE concepts::enable_if<
    type_traits::is_hip_policy<Pol>,
    RAJA::launch_is<Pol, RAJA::Launch::async> >
PreLaunch(resources::Hip & /*hip_res*/,
          Apollo::Region * /*region*/,
          Apollo::RegionContext *context)
{
  context->timer = Apollo::Timer::create<Apollo::Timer::HipAsync>();
  context->timer->start();
}
#endif

template <typename Pol, typename Res>
static RAJA_INLINE void PreLaunch(Res &,
                                  Apollo::Region *,
                                  Apollo::RegionContext *)
{
  // std::cout << "prelaunch default -> noop\n";
  //  default noop
}

template <typename Pol, typename Res>
static RAJA_INLINE void PostLaunch(Res &,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context)
{
  // std::cout << "postlaunch default -> calling region->end(); \n";
  region->end(context);
}

template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename BinFn>
struct InclusiveInplacePolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iter begin,
                                   Iter end,
                                   BinFn f)
  {
    using ExecutionPolicy = camp::at_v<PolicyList, idx>;

    if (policy == idx) {
      // generate policy variant, calls top-level forall pattern.
      auto r = resources::get_resource<ExecutionPolicy>::type::get_default();

      PreLaunch<ExecutionPolicy>(r, region, context);

      RAJA::impl::scan::inclusive_inplace(
          r, ExecutionPolicy(), std::move(begin), std::move(end), std::move(f));

      PostLaunch<ExecutionPolicy>(r, region, context);
    } else
      InclusiveInplacePolicyGeneratorSingle<idx + 1,
                                            num_policies,
                                            PolicyList,
                                            Iter,
                                            BinFn>::generate(policy,
                                                             region,
                                                             context,
                                                             std::move(begin),
                                                             std::move(end),
                                                             std::move(f));
  }
};

template <camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename BinFn>
struct InclusiveInplacePolicyGeneratorSingle<num_policies,
                                             num_policies,
                                             PolicyList,
                                             Iter,
                                             BinFn> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   Iter,
                                   Iter,
                                   BinFn)
  {
  }
};

template <typename PolicyList, typename Iter, typename BinFn>
static RAJA_INLINE void InclusiveInplacePolicyGenerator(
    int policy,
    Apollo::Region *region,
    Apollo::RegionContext *context,
    Iter begin,
    Iter end,
    BinFn f)
{
  InclusiveInplacePolicyGeneratorSingle<0,
                                        camp::size<PolicyList>::value,
                                        PolicyList,
                                        Iter,
                                        BinFn>::generate(policy,
                                                         region,
                                                         context,
                                                         std::move(begin),
                                                         std::move(end),
                                                         std::move(f));
}

template <typename Res, typename PolicyList, typename Iter, typename BinFn>
RAJA_INLINE Res inclusive_inplace(
    // concepts::enable_if<type_traits::is_apollo_policy<Policy>>
    Res r,
    const apollo_multi_exec<PolicyList> &,
    Iter begin,
    Iter end,
    BinFn f)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    apolloRegion =
        new Apollo::Region(/* num features */ 1,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<PolicyList>::value);
  }

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(begin, end);

  Apollo::RegionContext *context = apolloRegion->begin({num_elements});

  policy_index = apolloRegion->getPolicyIndex(context);

  InclusiveInplacePolicyGenerator<PolicyList, Iter, BinFn>(policy_index,
                                                           apolloRegion,
                                                           context,
                                                           std::move(begin),
                                                           std::move(end),
                                                           std::move(f));
  return {r};
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename BinFn,
          typename ValueT>
struct ExclusiveInplacePolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iter begin,
                                   Iter end,
                                   BinFn f,
                                   ValueT v)
  {
    using ExecutionPolicy = camp::at_v<PolicyList, idx>;

    if (policy == idx) {
      // generate policy variant, calls top-level forall pattern.
      auto r = resources::get_resource<ExecutionPolicy>::type::get_default();

      PreLaunch<ExecutionPolicy>(r, region, context);

      RAJA::impl::scan::exclusive_inplace(r,
                                          ExecutionPolicy(),
                                          std::move(begin),
                                          std::move(end),
                                          std::move(f),
                                          std::move(v));

      PostLaunch<ExecutionPolicy>(r, region, context);
    } else
      ExclusiveInplacePolicyGeneratorSingle<idx + 1,
                                            num_policies,
                                            PolicyList,
                                            Iter,
                                            BinFn,
                                            ValueT>::generate(policy,
                                                              region,
                                                              context,
                                                              std::move(begin),
                                                              std::move(end),
                                                              std::move(f),
                                                              std::move(v));
  }
};

template <camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename BinFn,
          typename ValueT>
struct ExclusiveInplacePolicyGeneratorSingle<num_policies,
                                             num_policies,
                                             PolicyList,
                                             Iter,
                                             BinFn,
                                             ValueT> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   Iter,
                                   Iter,
                                   BinFn,
                                   ValueT)
  {
  }
};

template <typename PolicyList, typename Iter, typename BinFn, typename ValueT>
static RAJA_INLINE void ExclusiveInplacePolicyGenerator(
    int policy,
    Apollo::Region *region,
    Apollo::RegionContext *context,
    Iter begin,
    Iter end,
    BinFn f,
    ValueT v)
{
  ExclusiveInplacePolicyGeneratorSingle<0,
                                        camp::size<PolicyList>::value,
                                        PolicyList,
                                        Iter,
                                        BinFn,
                                        ValueT>::generate(policy,
                                                          region,
                                                          context,
                                                          std::move(begin),
                                                          std::move(end),
                                                          std::move(f),
                                                          std::move(v));
}

template <typename Res,
          typename PolicyList,
          typename Iter,
          typename BinFn,
          typename ValueT>
RAJA_INLINE Res exclusive_inplace(Res r,
                                  const apollo_multi_exec<PolicyList> &,
                                  Iter begin,
                                  Iter end,
                                  BinFn f,
                                  ValueT v)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    apolloRegion =
        new Apollo::Region(/* num features */ 1,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<PolicyList>::value);
  }

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(begin, end);

  Apollo::RegionContext *context = apolloRegion->begin({num_elements});

  policy_index = apolloRegion->getPolicyIndex(context);

  ExclusiveInplacePolicyGenerator<PolicyList, Iter, BinFn, ValueT>(
      policy_index,
      apolloRegion,
      context,
      std::move(begin),
      std::move(end),
      std::move(f),
      std::move(v));
  return {r};
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn>
struct InclusivePolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iter begin,
                                   Iter end,
                                   OutIter out,
                                   BinFn f)
  {
    using ExecutionPolicy = camp::at_v<PolicyList, idx>;

    if (policy == idx) {
      // generate policy variant, calls top-level forall pattern.
      auto r = resources::get_resource<ExecutionPolicy>::type::get_default();

      PreLaunch<ExecutionPolicy>(r, region, context);

      RAJA::impl::scan::inclusive(r,
                                  ExecutionPolicy(),
                                  std::move(begin),
                                  std::move(end),
                                  std::move(out),
                                  std::move(f));

      PostLaunch<ExecutionPolicy>(r, region, context);
    } else
      InclusivePolicyGeneratorSingle<idx + 1,
                                     num_policies,
                                     PolicyList,
                                     Iter,
                                     OutIter,
                                     BinFn>::generate(policy,
                                                      region,
                                                      context,
                                                      std::move(begin),
                                                      std::move(end),
                                                      std::move(out),
                                                      std::move(f));
  }
};

template <camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn>
struct InclusivePolicyGeneratorSingle<num_policies,
                                      num_policies,
                                      PolicyList,
                                      Iter,
                                      OutIter,
                                      BinFn> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   Iter,
                                   Iter,
                                   OutIter,
                                   BinFn)
  {
  }
};

template <typename PolicyList, typename Iter, typename OutIter, typename BinFn>
static RAJA_INLINE void InclusivePolicyGenerator(int policy,
                                                 Apollo::Region *region,
                                                 Apollo::RegionContext *context,
                                                 Iter begin,
                                                 Iter end,
                                                 OutIter out,
                                                 BinFn f)
{
  InclusivePolicyGeneratorSingle<0,
                                 camp::size<PolicyList>::value,
                                 PolicyList,
                                 Iter,
                                 OutIter,
                                 BinFn>::generate(policy,
                                                  region,
                                                  context,
                                                  std::move(begin),
                                                  std::move(end),
                                                  std::move(out),
                                                  std::move(f));
}

template <typename Res,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn>
Res inclusive(Res r,
              const apollo_multi_exec<PolicyList> &,
              Iter begin,
              Iter end,
              OutIter out,
              BinFn f)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    apolloRegion =
        new Apollo::Region(/* num features */ 1,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<PolicyList>::value);
  }

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(begin, end);

  Apollo::RegionContext *context = apolloRegion->begin({num_elements});

  policy_index = apolloRegion->getPolicyIndex(context);

  InclusivePolicyGenerator<PolicyList, Iter, OutIter, BinFn>(policy_index,
                                                             apolloRegion,
                                                             context,
                                                             std::move(begin),
                                                             std::move(end),
                                                             std::move(out),
                                                             std::move(f));
  return {r};
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/

template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
struct ExclusivePolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Iter begin,
                                   Iter end,
                                   OutIter out,
                                   BinFn f,
                                   ValueT v)
  {
    using ExecutionPolicy = camp::at_v<PolicyList, idx>;

    if (policy == idx) {
      // generate policy variant, calls top-level forall pattern.
      auto r = resources::get_resource<ExecutionPolicy>::type::get_default();

      PreLaunch<ExecutionPolicy>(r, region, context);

      RAJA::impl::scan::exclusive(r,
                                  ExecutionPolicy(),
                                  std::move(begin),
                                  std::move(end),
                                  std::move(out),
                                  std::move(f),
                                  std::move(v));

      PostLaunch<ExecutionPolicy>(r, region, context);
    } else
      ExclusivePolicyGeneratorSingle<idx + 1,
                                     num_policies,
                                     PolicyList,
                                     Iter,
                                     OutIter,
                                     BinFn,
                                     ValueT>::generate(policy,
                                                       region,
                                                       context,
                                                       std::move(begin),
                                                       std::move(end),
                                                       std::move(out),
                                                       std::move(f),
                                                       std::move(v));
  }
};

template <camp::idx_t num_policies,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
struct ExclusivePolicyGeneratorSingle<num_policies,
                                      num_policies,
                                      PolicyList,
                                      Iter,
                                      OutIter,
                                      BinFn,
                                      ValueT> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   Iter,
                                   Iter,
                                   OutIter,
                                   BinFn,
                                   ValueT)
  {
  }
};

template <typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
static RAJA_INLINE void ExclusivePolicyGenerator(int policy,
                                                 Apollo::Region *region,
                                                 Apollo::RegionContext *context,
                                                 Iter begin,
                                                 Iter end,
                                                 OutIter out,
                                                 BinFn f,
                                                 ValueT v)
{
  ExclusivePolicyGeneratorSingle<0,
                                 camp::size<PolicyList>::value,
                                 PolicyList,
                                 Iter,
                                 OutIter,
                                 BinFn,
                                 ValueT>::generate(policy,
                                                   region,
                                                   context,
                                                   std::move(begin),
                                                   std::move(end),
                                                   std::move(out),
                                                   std::move(f),
                                                   std::move(v));
}

template <typename Res,
          typename PolicyList,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
Res exclusive(Res r,
              const apollo_multi_exec<PolicyList> &,
              Iter begin,
              Iter end,
              OutIter out,
              BinFn f,
              ValueT v)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;
  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    apolloRegion =
        new Apollo::Region(/* num features */ 1,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<PolicyList>::value);
  }

  // Count the number of elements.
  float num_elements = 0.0;
  num_elements = (float)std::distance(begin, end);

  Apollo::RegionContext *context = apolloRegion->begin({num_elements});

  policy_index = apolloRegion->getPolicyIndex(context);

  ExclusivePolicyGenerator<PolicyList, Iter, OutIter, BinFn, ValueT>(
      policy_index,
      apolloRegion,
      context,
      std::move(begin),
      std::move(end),
      std::move(out),
      std::move(f),
      std::move(v));
  return {r};
}


/// Range implementation.

template <typename Res,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async,
          typename Iter,
          typename BinFn>
RAJA_INLINE Res
inclusive_inplace(Res r,
                  const hip_exec_apollo_runtime<BLOCK_SIZE_START,
                                                BLOCK_SIZE_END,
                                                BLOCK_SIZE_STEP,
                                                Async> &,
                  Iter begin,
                  Iter end,
                  BinFn f)
{
  RAJA::impl::scan::inclusive_inplace(
      r, hip_exec<0, Async>(), std::move(begin), std::move(end), std::move(f));

  return {r};
}

template <typename Res,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async,
          typename Iter,
          typename OutIter,
          typename BinFn>
RAJA_INLINE Res
inclusive(Res r,
                  const hip_exec_apollo_runtime<BLOCK_SIZE_START,
                                                BLOCK_SIZE_END,
                                                BLOCK_SIZE_STEP,
                                                Async> &,
                  Iter begin,
                  Iter end,
                  OutIter out,
                  BinFn f)
{
  RAJA::impl::scan::inclusive(r,
                              hip_exec<0, Async>(),
                              std::move(begin),
                              std::move(end),
                              std::move(out),
                              std::move(f));

  return {r};
}

template <typename Res,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async,
          typename Iter,
          typename BinFn,
          typename ValueT>
RAJA_INLINE Res
exclusive_inplace(Res r,
                  const hip_exec_apollo_runtime<BLOCK_SIZE_START,
                                                BLOCK_SIZE_END,
                                                BLOCK_SIZE_STEP,
                                                Async> &,

                  Iter begin,
                  Iter end,
                  BinFn f,
                  ValueT v)
{
  RAJA::impl::scan::exclusive_inplace(r,
                                      hip_exec<0, Async>(),
                                      std::move(begin),
                                      std::move(end),
                                      std::move(f),
                                      std::move(v));
  return {r};
}

template <typename Res,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          bool Async,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
Res exclusive(Res r,
              const hip_exec_apollo_runtime<BLOCK_SIZE_START,
                                            BLOCK_SIZE_END,
                                            BLOCK_SIZE_STEP,
                                            Async> &,
              Iter begin,
              Iter end,
              OutIter out,
              BinFn f,
              ValueT v)
{
  RAJA::impl::scan::exclusive(r,
                              hip_exec<0, Async>(),
                              std::move(begin),
                              std::move(end),
                              std::move(out),
                              std::move(f),
                              std::move(v));

  return {r};
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
