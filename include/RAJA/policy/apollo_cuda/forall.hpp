/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA segment template methods for
 *          execution via CUDA kernel launch.
 *
 *          These methods should work on any platform that supports
 *          CUDA devices.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_forall_apollo_cuda_HPP
#define RAJA_forall_apollo_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_CUDA) && defined(RAJA_ENABLE_APOLLO)

#include <algorithm>

#include "RAJA/pattern/forall.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/internal/fault_tolerance.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/apollo_cuda/policy.hpp"
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"

#include "RAJA/index/IndexSet.hpp"

#include "RAJA/util/resource.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA
{

namespace policy
{

namespace cuda
{

namespace impl
{

template <typename Iterator,
          typename LOOP_BODY,
          typename IndexType>
    __global__
    void forall_apollo_cuda_kernel(LOOP_BODY loop_body,
                            const Iterator idx,
                            IndexType length)
{
  using RAJA::internal::thread_privatize;
  auto privatizer = thread_privatize(loop_body);
  auto& body = privatizer.get_priv();
  auto ii = static_cast<IndexType>(getGlobalIdx_1D_1D());
  if (ii < length) {
    body(idx[ii]);
  }
}

}  // namespace impl

//
////////////////////////////////////////////////////////////////////////
//
// Function templates for CUDA execution over iterables.
//
////////////////////////////////////////////////////////////////////////
//

struct ApolloCallbackHelper {
  struct callback_t {
    Apollo::Apollo *apollo;
    Apollo::Region *region;
    Apollo::RegionContext *context;
  };

  static void callbackFunction(void *data)
  {
    callback_t *cbdata = reinterpret_cast<callback_t *>(data);
    cbdata->region->end(cbdata->context);
    delete cbdata;
  }
};

template <typename Iterable, typename LoopBody, size_t BlockSize, bool Async>
RAJA_INLINE resources::EventProxy<resources::Cuda> forall_impl(resources::Cuda &cuda_res,
                                                    apollo_cuda_exec<BlockSize, Async>,
                                                    Iterable&& iter,
                                                    LoopBody&& loop_body)
{
  using Iterator  = camp::decay<decltype(std::begin(iter))>;
  using LOOP_BODY = camp::decay<LoopBody>;
  using IndexType = camp::decay<decltype(std::distance(std::begin(iter), std::end(iter)))>;

  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;

  cudaStream_t stream = cuda_res.get_stream();

  //
  // Compute the requested iteration space size
  //
  Iterator begin = std::begin(iter);
  Iterator end = std::end(iter);
  IndexType len = std::distance(begin, end);

  // Only launch kernel if we have something to iterate over
  if (len > 0 && BlockSize > 0) {
    auto func = impl::forall_apollo_cuda_kernel<Iterator, LOOP_BODY, IndexType>;

    int policy_index = 0;
    static int ApolloBlockSize_policies;
    static std::vector<float> func_features;
    if (apolloRegion == nullptr) {
        // one-time initialization
        std::string code_location = apollo->getCallpathOffset();
        ApolloBlockSize_policies = 4;
        cudaFuncAttributes attr;
        cudaFuncGetAttributes(&attr, func);

        // TODO: remove printout comments
        /*std::cout << "constSizeBytes " << attr.constSizeBytes << std::endl;
        std::cout << "localSizeBytes " << attr.localSizeBytes << std::endl;
        std::cout << "numRegs " << attr.numRegs << std::endl;
        std::cout << "sharedSizeBytes " << attr.sharedSizeBytes << std::endl;*/

        func_features = {static_cast<float>(attr.constSizeBytes),
                         static_cast<float>(attr.localSizeBytes),
                         static_cast<float>(attr.numRegs),
                         static_cast<float>(attr.sharedSizeBytes)};

        apolloRegion = new Apollo::Region(1 + func_features.size(),
                                          code_location.c_str(),
                                          ApolloBlockSize_policies);
    }

    ApolloCallbackHelper::callback_t *cbdata =
        new ApolloCallbackHelper::callback_t();
    cbdata->apollo = apollo;
    cbdata->region = apolloRegion;

    std::vector<float> features( { static_cast<float>(len) });
    features.insert( features.begin(), func_features.begin(), func_features.end() );
    cbdata->context = apolloRegion->begin( features );

    policy_index = apolloRegion->getPolicyIndex(cbdata->context);

    cuda_dim_member_t ApolloBlockSize = BlockSize / ( 1 << (policy_index) );
    ApolloBlockSize = (ApolloBlockSize > 1) ? ApolloBlockSize : 1;

    //
    // Compute the number of blocks
    //
    cuda_dim_t blockSize{ApolloBlockSize, 1, 1};
    cuda_dim_t gridSize = impl::getGridDim(static_cast<cuda_dim_member_t>(len), blockSize);

    RAJA_FT_BEGIN;

    //
    // Setup shared memory buffers
    //
    size_t shmem = 0;

    //  printf("gridsize = (%d,%d), blocksize = %d\n",
    //         (int)gridSize.x,
    //         (int)gridSize.y,
    //         (int)blockSize.x);

    {
      //
      // Privatize the loop_body, using make_launch_body to setup reductions
      //
      //std::cout << "blockSize " << blockSize.x << ", " << blockSize.y << ", " << blockSize.z
      //<< " | gridSize" << gridSize.x << ", " << gridSize.y << ", " << gridSize.z << std::endl;
      LOOP_BODY body = RAJA::cuda::make_launch_body(
          gridSize, blockSize, shmem, stream, std::forward<LoopBody>(loop_body));

      //
      // Launch the kernels
      //
      void *args[] = {(void*)&body, (void*)&begin, (void*)&len};
      RAJA::cuda::launch((const void*)func, gridSize, blockSize, args, shmem, stream);
      cudaLaunchHostFunc(stream, ApolloCallbackHelper::callbackFunction, cbdata);
    }

    if (!Async) { RAJA::cuda::synchronize(stream); }

    RAJA_FT_END;
  }

  return resources::EventProxy<resources::Cuda>(&cuda_res);
}


//
//////////////////////////////////////////////////////////////////////
//
// The following function templates iterate over index set segments
// using the explicitly named segment iteration policy and execute
// segments as CUDA kernels.
//
//////////////////////////////////////////////////////////////////////
//

/*!
 ******************************************************************************
 *
 * \brief  Sequential iteration over segments of index set and
 *         CUDA execution for segments.
 *
 ******************************************************************************
 */
template <typename LoopBody,
          size_t BlockSize,
          bool Async,
          typename... SegmentTypes>
RAJA_INLINE resources::EventProxy<resources::Cuda> forall_impl(resources::Cuda &r,
                                                    ExecPolicy<seq_segit, apollo_cuda_exec<BlockSize, Async>>,
                                                    const TypedIndexSet<SegmentTypes...>& iset,
                                                    LoopBody&& loop_body)
{
  int num_seg = iset.getNumSegments();
  for (int isi = 0; isi < num_seg; ++isi) {
    iset.segmentCall(r,
                     isi,
                     detail::CallForall(),
                     apollo_cuda_exec<BlockSize, true>(),
                     loop_body);
  }  // iterate over segments of index set

  if (!Async) RAJA::cuda::synchronize();
  return resources::EventProxy<resources::Cuda>(&r);
}

}  // namespace apollo_cuda

}  // namespace policy

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
