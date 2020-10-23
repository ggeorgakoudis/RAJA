/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_apollo_cuda_kernel_CudaKernel_HPP
#define RAJA_policy_apollo_cuda_kernel_CudaKernel_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_APOLLO)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/policy/cuda/MemUtils_CUDA.hpp"
#include "RAJA/policy/cuda/policy.hpp"

#include "RAJA/policy/cuda/kernel/internal.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA
{

namespace statement
{

/*!
 * A RAJA::kernel statement that launches a CUDA kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct ApolloCudaKernelExt
    : public internal::Statement<cuda_exec<0>, EnclosedStmts...> {
};

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * The kernel launch is synchronous.
 */
template <size_t num_threads, typename... EnclosedStmts>
using ApolloCudaKernelFixed =
    ApolloCudaKernelExt<cuda_explicit_launch<false, operators::limits<size_t>::max(), num_threads>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with a fixed
 * number of threads (specified by num_threads)
 * The kernel launch is asynchronous.
 */
template <size_t num_threads, typename... EnclosedStmts>
using ApolloCudaKernelFixedAsync =
    ApolloCudaKernelExt<cuda_explicit_launch<true, operators::limits<size_t>::max(), num_threads>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with 1024 threads
 * The kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using ApolloCudaKernel = ApolloCudaKernelFixed<1024, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a CUDA kernel with 1024 threads
 * The kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using ApolloCudaKernelAsync = ApolloCudaKernelFixedAsync<1024, EnclosedStmts...>;

}  // namespace statement

namespace internal
{

/*!
 * Helper class that handles CUDA kernel launching, and computing
 * maximum number of threads/blocks
 */
template<typename LaunchPolicy, typename StmtList, typename Data, typename Types>
struct ApolloCudaLaunchHelper;


/*!
 * Helper class specialization to determine the number of threads and blocks.
 * The user may specify the number of threads and blocks or let one or both be
 * determined at runtime using the CUDA occupancy calculator.
 */
template<bool async0, size_t num_blocks, size_t num_threads, typename StmtList, typename Data, typename Types>
struct ApolloCudaLaunchHelper<cuda_launch<async0, num_blocks, num_threads>,StmtList,Data,Types>
{
  using Self = ApolloCudaLaunchHelper;

  static constexpr bool async = async0;

  using executor_t = internal::cuda_statement_list_executor_t<StmtList, Data, Types>;

  using kernelGetter_t = CudaKernelLauncherGetter<(num_threads <= 0) ? 0 : num_threads, Data, executor_t>;

  inline static void recommended_blocks_threads(int shmem_size,
      size_t &recommended_blocks, size_t &recommended_threads)
  {
    auto func = kernelGetter_t::get();

    if (num_blocks <= 0) {

      if (num_threads <= 0) {

        //
        // determine blocks at runtime
        // determine threads at runtime
        //
        internal::cuda_occupancy_max_blocks_threads<Self>(
            func, shmem_size, recommended_blocks, recommended_threads);

      } else {

        //
        // determine blocks at runtime
        // threads determined at compile-time
        //
        recommended_threads = num_threads;

        internal::cuda_occupancy_max_blocks<Self, num_threads>(
            func, shmem_size, recommended_blocks);

      }

    } else {

      if (num_threads <= 0) {

        //
        // determine threads at runtime, unsure what use 1024
        // this value may be invalid for kernels with high register pressure
        //
        recommended_threads = 1024;

      } else {

        //
        // threads determined at compile-time
        //
        recommended_threads = num_threads;

      }

      //
      // blocks determined at compile-time
      //
      recommended_blocks = num_blocks;

    }
  }

  inline static void max_threads(int RAJA_UNUSED_ARG(shmem_size), size_t &max_threads)
  {
    if (num_threads <= 0) {

      //
      // determine threads at runtime, unsure what use 1024
      // this value may be invalid for kernels with high register pressure
      //
      max_threads = 1024;

    } else {

      //
      // threads determined at compile-time
      //
      max_threads = num_threads;

    }
  }

  inline static void max_blocks(int shmem_size,
      size_t &max_blocks, size_t actual_threads)
  {
    auto func = kernelGetter_t::get();

    if (num_blocks <= 0) {

      //
      // determine blocks at runtime
      //
      if (num_threads <= 0 ||
          num_threads != actual_threads) {

        //
        // determine blocks when actual_threads != num_threads
        //
        internal::cuda_occupancy_max_blocks<Self>(
            func, shmem_size, max_blocks, actual_threads);

      } else {

        //
        // determine blocks when actual_threads == num_threads
        //
        internal::cuda_occupancy_max_blocks<Self, num_threads>(
            func, shmem_size, max_blocks);

      }

    } else {

      //
      // blocks determined at compile-time
      //
      max_blocks = num_blocks;

    }

  }

  static void launch(Data &&data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     cudaStream_t stream)
  {
    auto func = kernelGetter_t::get();

    void *args[] = {(void*)&data};
    RAJA::cuda::launch((const void*)func, launch_dims.blocks, launch_dims.threads, args, shmem, stream);
  }
};

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

/*!
 * Specialization that launches CUDA kernels for RAJA::kernel from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts, typename Types>
struct StatementExecutor<
    statement::ApolloCudaKernelExt<LaunchConfig, EnclosedStmts...>, Types> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::ApolloCudaKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static inline void exec(Data &&data)
  {

    using data_t = camp::decay<Data>;
    using executor_t = cuda_statement_list_executor_t<stmt_list_t, data_t, Types>;
    using launch_t = ApolloCudaLaunchHelper<LaunchConfig, stmt_list_t, data_t, Types>;

    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);

    std::vector<float> *features = new std::vector<float>();
    executor_t::getFeatures(data, *features);

    // Only launch kernel if we have something to iterate over
    size_t num_blocks = launch_dims.num_blocks();
    size_t num_threads = launch_dims.num_threads();
    if (num_blocks > 0 || num_threads > 0) {

      static Apollo *apollo = Apollo::instance();
      static Apollo::Region *apolloRegion = nullptr;

      //
      // Setup shared memory buffers
      //
      int shmem = 0;
      cudaStream_t stream = 0;

      //
      // Compute the MAX physical kernel threads
      //
      size_t max_threads;
      launch_t::max_threads(shmem, max_threads);

      auto getLaunchDims = [&](size_t num_blocks, size_t num_threads) {
        //
        // Fit the requested threads
        //
        cuda_dim_t fit_threads{0, 0, 0};

        //fit_threads = fitCudaDims(max_threads,
        fit_threads =
            fitCudaDims(num_threads < max_threads ? num_threads : max_threads,
                        launch_dims.threads,
                        launch_dims.min_threads);

        launch_dims.threads = fit_threads;

        launch_dims.blocks =
            fitCudaDims(num_blocks, launch_dims.blocks, launch_dims.min_blocks);
      };

      {
        //
        // Privatize the LoopData, using make_launch_body to setup reductions
        //
        auto cuda_data = RAJA::cuda::make_launch_body(
            launch_dims.blocks, launch_dims.threads, shmem, stream, data);


        //
        // Launch the kernels
        //

        int policy_index = 0;
        static int num_blocks_policies;
        static int num_threads_policies;
        if (apolloRegion == nullptr) {
            // one-time initialization
            std::string code_location = apollo->getCallpathOffset();
            //num_blocks_policies = std::ceil(std::log2(launch_dims.num_blocks()));
            num_blocks_policies = 4;
            num_threads_policies = 4;
            apolloRegion =
                new Apollo::Region(features->size(),
                                   code_location.c_str(),
                                   num_blocks_policies * num_threads_policies);
        }

        ApolloCallbackHelper::callback_t *cbdata = new ApolloCallbackHelper::callback_t();
        cbdata->apollo = apollo;
        cbdata->region = apolloRegion;
        cbdata->context = apolloRegion->begin( *features );
        delete features;
        /*
        cbdata->context = apolloRegion->begin(
            {(float)launch_dims.blocks.x,
             (float)launch_dims.blocks.y,
             (float)launch_dims.blocks.z,
             (float)launch_dims.threads.x,
             (float)launch_dims.threads.y,
             (float)launch_dims.threads.z});
        */

        policy_index = apolloRegion->getPolicyIndex(cbdata->context);

        // Start from high block sizes to smaller.
        size_t num_blocks = launch_dims.num_blocks() / ( 1 << (policy_index%num_blocks_policies) );

        size_t num_threads = max_threads / ( 1 << (policy_index%num_threads_policies) );

        //std::cout << "policy " << policy_index << " / "
        //          << num_blocks_policies * num_threads_policies
        //          << " num_blocks " << num_blocks << " num_threads "
        //          << num_threads << std::endl;
        getLaunchDims(num_blocks, num_threads);

        launch_t::launch(std::move(cuda_data), launch_dims, shmem, stream);
        cudaLaunchHostFunc(stream, ApolloCallbackHelper::callbackFunction, cbdata);
      }

      //
      // Synchronize
      //
      if (!launch_t::async) { RAJA::cuda::synchronize(stream); }
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
