/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing user interface for
 *RAJA::launch::apollo_multi
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_launch_cuda_HPP
#define RAJA_pattern_launch_cuda_HPP

#include "RAJA/pattern/launch/launch_core.hpp"
#include "RAJA/pattern/detail/privatizer.hpp"
#include "RAJA/policy/apollo_multi/policy.hpp"
#include "RAJA/util/resource.hpp"

namespace RAJA
{

namespace apollo
{

namespace launch
{

template <bool Async = false>
void PreLaunch(Apollo::RegionContext *)
{
}

template <>
void PreLaunch<true>(Apollo::RegionContext *context)
{
  context->timer = Apollo::Timer::create<Apollo::Timer::HipAsync>();
  context->timer->start();
}

} // end namespace launch

} // end namespace apollo

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP>
struct LaunchExecute<RAJA::hip_launch_apollo_range_t<Async,
                                        GRID_SIZE_START,
                                        GRID_SIZE_END,
                                        GRID_SIZE_STEP,
                                        BLOCK_SIZE_START,
                                        BLOCK_SIZE_END,
                                        BLOCK_SIZE_STEP>> {

  template <typename BODY_IN>
  static resources::EventProxy<resources::Resource>
  exec(RAJA::resources::Resource res, const LaunchParams &in_params, const char *kernel_name, BODY_IN &&body_in)
  {

    static Apollo *apollo = Apollo::instance();
    static Apollo::Region *apolloRegion = nullptr;
    static int policy_index = 0;

    constexpr size_t num_of_grid_sizes = 1 + (GRID_SIZE_END - GRID_SIZE_START) / GRID_SIZE_STEP;
    constexpr size_t num_of_block_sizes = 1 + (BLOCK_SIZE_END - BLOCK_SIZE_START) / BLOCK_SIZE_STEP;
    constexpr size_t num_policies = num_of_grid_sizes * num_of_block_sizes;

    if (apolloRegion == nullptr) {
      std::string code_location = apollo->getCallpathOffset();
      apolloRegion = new Apollo::Region(
          /* num features */ 0,
          /* region id */ code_location.c_str(),
          /* num policies */ num_policies);
    }

    std::vector<float> features;
    //FeatureGenerator<SegmentTuple>::generate(segments, features);
    /*std::cout << "features: [ ";
    for(auto e : features)
      std::cout << e << ", ";
    std::cout << " ]\n";*/

    Apollo::RegionContext *context = apolloRegion->begin(features);

    policy_index = apolloRegion->getPolicyIndex(context);

    // Iterate policies.
    int policy_index_grid_size = policy_index / num_of_block_sizes;
    int policy_index_block_size = policy_index % num_of_block_sizes;

    size_t GridSize = GRID_SIZE_START + policy_index_grid_size * GRID_SIZE_STEP;
    size_t BlockSize =
        BLOCK_SIZE_START + policy_index_block_size * BLOCK_SIZE_STEP;

    LaunchParams params(Teams(GridSize),
                        Threads(BlockSize),
                        in_params.shared_mem_size);

    apollo::launch::PreLaunch<Async>(context);

    LaunchExecute<hip_launch_t<Async>>::exec(res,
                                             params,
                                             kernel_name,
                                             std::forward<BODY_IN>(body_in));

    apolloRegion->end(context);

    return resources::EventProxy<resources::Resource>(res);
  }
};

}  // namespace RAJA
#endif
