/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for apollo_multi execution.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_apollo_multi_HPP
#define RAJA_apollo_multi_HPP

#if defined(RAJA_ENABLE_APOLLO)

#include "apollo/Apollo.h"
#include "apollo/Region.h"

#if defined(RAJA_CUDA_ACTIVE)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#if defined(RAJA_HIP_ACTIVE)
#include <hip_runtime.h>
#endif

namespace RAJA {

namespace apollo {

struct ApolloCallbackDataPool : Apollo::CallbackDataPool {

struct callback_t {
    ApolloCallbackDataPool *data_pool;
#if defined(RAJA_CUDA_ACTIVE)
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

  void *get() override {
    if(pool.empty())
      expand(expand_size);

    callback_t *cbdata = pool.front();
    pool.pop_front();
    return cbdata;
  }

  void put(void *data) override {
    callback_t *cbdata = reinterpret_cast<callback_t *>(data);
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

#endif // RAJA_CUDA_ACTIVE

#include "RAJA/policy/apollo_multi/forall.hpp"
#include "RAJA/policy/apollo_multi/kernel.hpp"
//#include "RAJA/policy/apollo_multi/policy.hpp"
//#include "RAJA/policy/apollo_multi/scan.hpp"


#endif // RAJA_ENABLE_APOLLO

#endif  // closing endif for header file include guard
