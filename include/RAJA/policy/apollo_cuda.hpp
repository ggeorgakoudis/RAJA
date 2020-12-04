/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for NVCC CUDA execution.
 *
 *          These methods work only on platforms that support CUDA.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_apollo_cuda_HPP
#define RAJA_apollo_cuda_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_CUDA_ACTIVE)

#if defined(RAJA_ENABLE_APOLLO)

#include <cuda.h>
#include <cuda_runtime.h>

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA {

namespace apollo_cuda {

struct ApolloCallbackDataPool : Apollo::CallbackDataPool {

struct callback_t {
    ApolloCallbackDataPool *data_pool;
    cudaEvent_t start;
    cudaEvent_t stop;
  };

  std::list<callback_t *> pool;
  int expand_size;

  ApolloCallbackDataPool(int init_size, int expand_size)
      : expand_size(expand_size)
  {
    expand(init_size);
  }

  void expand(int size) {
    for (int i = 0; i < size; i++) {
      callback_t *cbdata = new callback_t;
      cudaEventCreateWithFlags(&cbdata->start, cudaEventDefault);
      cudaEventCreateWithFlags(&cbdata->stop, cudaEventDefault);
      pool.push_back(cbdata);
      cbdata->data_pool = this;
    }
  }

  callback_t *get() {
    if(pool.empty())
      expand(expand_size);

    callback_t *cbdata = pool.front();
    pool.pop_front();
    return cbdata;
  }

  void put(callback_t *cbdata) {
    pool.push_back(cbdata);
  }

  ~ApolloCallbackDataPool() {
    for(auto *cbdata : pool) {
      cudaEventDestroy(cbdata->start);
      cudaEventDestroy(cbdata->stop);
      delete cbdata;
    }
  }
};


struct ApolloCallbackHelper {
  static bool isDoneCallback(void *data, bool *returnsTime, double *time)
  {
    using callback_data_t = ApolloCallbackDataPool::callback_t;
    callback_data_t *cbdata = reinterpret_cast<callback_data_t *>(data);
    float cudaTime;

    *returnsTime = true;

    if (cudaEventElapsedTime(&cudaTime, cbdata->start, cbdata->stop) != cudaSuccess) {
      *time = 0.0;
      return false;
    }

    cbdata->data_pool->put(cbdata);

    // Convert from ms to s
    *time = cudaTime/1000.0;
    return true;
  }
};


}

}

//#include "RAJA/policy/cuda/atomic.hpp"
#include "RAJA/policy/apollo_cuda/forall.hpp"
#include "RAJA/policy/apollo_cuda/policy.hpp"
//#include "RAJA/policy/cuda/reduce.hpp"
#include "RAJA/policy/apollo_cuda/scan.hpp"
//#include "RAJA/policy/cuda/sort.hpp"
#include "RAJA/policy/apollo_cuda/kernel.hpp"
//#include "RAJA/policy/cuda/synchronize.hpp"
//#include "RAJA/policy/cuda/WorkGroup.hpp"

#endif // closing endif for if defined(RAJA_ENABLE_APOLLO)

#endif  // closing endif for if defined(RAJA_ENABLE_CUDA)

#endif  // closing endif for header file include guard
