//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "apollo/Apollo.h"
#include "RAJA/RAJA.hpp"

int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

  static constexpr std::size_t size = 1000000;
  static constexpr std::size_t iters = 1000;
  static constexpr std::size_t training_frequency = 10;

#if defined(RAJA_ENABLE_HIP)
  std::cout << "\n Running RAJA HIP daxpy with Apollo...\n";
  Apollo* apollo = Apollo::instance();

  using PolicyList = camp::list<
      RAJA::hip_exec<64>,
      RAJA::hip_exec<256>,
      RAJA::hip_exec<1024>>;

  double* a = nullptr; 
  double* b = nullptr;
  double c = 3.14;
  hipErrchk(hipMallocManaged( (void**)&a, size * sizeof(double) ));
  hipErrchk(hipMallocManaged( (void**)&b, size * sizeof(double) ));

  for (int i = 0; i < iters; i++) {
    size_t test_size = random() % size + 1;

    RAJA::forall<RAJA::apollo_multi_exec<PolicyList>>(
        RAJA::RangeSegment(0, test_size),
      [=] RAJA_DEVICE (int i) {
      a[i] += b[i] * c;
    });

    if (i % training_frequency == 0) {
      apollo->flushAllRegionMeasurements(i);
    }
  }


  hipErrchk(hipFree(a));
  hipErrchk(hipFree(b));
#endif
  
  std::cout << "\n DONE!...\n";

  return 0;
}
