//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "memoryManager.hpp"

#include "apollo/Apollo.h"
#include "RAJA/RAJA.hpp"

using namespace RAJA;

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{
  int iters = 20;
  for (int i = 0; i < iters; i++) {
    using LaunchPolicy = RAJA::LaunchPolicy<RAJA::hip_launch_apollo_range_t<false, 1,4,1, 64,256,64>>;
    using LoopPolicy = RAJA::LoopPolicy<hip_thread_x_loop>;
    RAJA::launch<LaunchPolicy>(
        RAJA::LaunchParams(RAJA::Teams(1), RAJA::Threads(1)),
        [=] RAJA_DEVICE (RAJA::LaunchContext &ctx) {
        RAJA::loop<LoopPolicy>(ctx, RAJA::RangeSegment(0, 1024), [&](int i) {
            printf("i %d hello!\n", i);
            });
        });
  }

  RAJA::synchronize<RAJA::hip_synchronize>();

  std::cout << "\n DONE!...\n";

  return 0;
}
