/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
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

#ifndef RAJA_policy_apollo_nested_HPP
#define RAJA_policy_apollo_nested_HPP

#include "RAJA/config.hpp"

#ifndef RAJA_ENABLE_OPENMP
#error "*** RAJA_ENABLE_OPENMP is not defined!" \
    "This build of RAJA requires OpenMP to be enabled! ***"
#endif

#if defined(RAJA_ENABLE_OPENMP)

#include <cassert>
#include <climits>

#include "RAJA/pattern/detail/privatizer.hpp"

#include "RAJA/pattern/kernel/Collapse.hpp"
#include "RAJA/pattern/kernel/internal.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/policy/apollo/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"

#if !defined(RAJA_COMPILER_MSVC)
#define RAJA_COLLAPSE(X) collapse(X)
#else
#define RAJA_COLLAPSE(X)
#endif

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA
{

struct apollo_collapse_exec
    : make_policy_pattern_t<RAJA::Policy::apollo,
                            RAJA::Pattern::forall,
                            RAJA::policy::omp::For> {
};

namespace internal
{

/////////
// Collapsing two loops
/////////

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<apollo_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
      static Apollo         *apollo             = Apollo::instance();
      static Apollo::Region *apolloRegion       = nullptr;
      static int             policy_index       = 0;
      using RAJA::policy::apollo::POLICY_COUNT;
      static int             num_threads[POLICY_COUNT] = { 0 };
      if (apolloRegion == nullptr) {
          std::string code_location = apollo->getCallpathOffset();
          apolloRegion = new Apollo::Region(
                  2, //num features
                  code_location.c_str(), // region uid
                  POLICY_COUNT // num policies
                  );
          // Set the range of thread counts we want to make available for
          // bootstrapping and use by this Apollo::Region.
          num_threads[0] = apollo->ompDefaultNumThreads;
          num_threads[1] = 1;

          num_threads[2] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[3] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[4] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[5] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[6] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[7] = 2;
          num_threads[8] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[9] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[10] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[11] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[12] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[13] = 2;
          num_threads[14] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[15] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[16] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[17] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[18] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[19] = 2;
      }
      const auto l0 = segment_length<Arg0>(data);
      const auto l1 = segment_length<Arg1>(data);
      // NOTE: these are here to avoid a use-after-scope detected by address
      // sanitizer, probably a false positive, but the result should be
      // essentially identical
      auto i0 = l0;
      auto i1 = l1;

      using RAJA::internal::thread_privatize;
      auto privatizer = thread_privatize(data);

      apolloRegion->begin();

      apolloRegion->setFeature(l0);
      apolloRegion->setFeature(l1);

      policy_index = apolloRegion->getPolicyIndex();

      switch(policy_index) {
          case 0:
              {
                  //std::cout << "2-level OMP defaults" << std::endl; //ggout
#pragma omp parallel for private(i0, i1) firstprivate(privatizer) \
                  RAJA_COLLAPSE(2)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          auto& private_data = privatizer.get_priv();
                          private_data.template assign_offset<Arg0>(i0);
                          private_data.template assign_offset<Arg1>(i1);
                          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                      }
                  }
                  break;
              }
          case 1:
              {
                  //std::cout << "Sequential" << std::endl; //ggout
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          auto& private_data = privatizer.get_priv();
                          private_data.template assign_offset<Arg0>(i0);
                          private_data.template assign_offset<Arg1>(i1);
                          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                      }
                  }
                  break;
              }
          case 2:
          case 3:
          case 4:
          case 5:
          case 6:
          case 7:
              {
                  //std::cout << "OMP static num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(static) \
                  RAJA_COLLAPSE(2)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          auto& private_data = privatizer.get_priv();
                          private_data.template assign_offset<Arg0>(i0);
                          private_data.template assign_offset<Arg1>(i1);
                          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                      }
                  }
                  break;
              }
          case 8:
          case 9:
          case 10:
          case 11:
          case 12:
          case 13:
              {
                  //std::cout << "OMP dynamic num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(dynamic) \
                  RAJA_COLLAPSE(2)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          auto& private_data = privatizer.get_priv();
                          private_data.template assign_offset<Arg0>(i0);
                          private_data.template assign_offset<Arg1>(i1);
                          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                      }
                  }
                  break;
              }
          case 14:
          case 15:
          case 16:
          case 17:
          case 18:
          case 19:
              {
                  //std::cout << "OMP guided num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(guided) \
                  RAJA_COLLAPSE(2)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          auto& private_data = privatizer.get_priv();
                          private_data.template assign_offset<Arg0>(i0);
                          private_data.template assign_offset<Arg1>(i1);
                          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                      }
                  }
                  break;
              }
      }

      apolloRegion->end();
  }
};


template <camp::idx_t Arg0,
          camp::idx_t Arg1,
          camp::idx_t Arg2,
          typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<apollo_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
      static Apollo         *apollo             = Apollo::instance();
      static Apollo::Region *apolloRegion       = nullptr;
      static int             policy_index       = 0;
      using RAJA::policy::apollo::POLICY_COUNT;
      static int             num_threads[POLICY_COUNT] = { 0 };
      if (apolloRegion == nullptr) {
          std::string code_location = apollo->getCallpathOffset();
          apolloRegion = new Apollo::Region(
                  3, //num features
                  code_location.c_str(), // region uid
                  POLICY_COUNT // num policies
                  );
          // Set the range of thread counts we want to make available for
          // bootstrapping and use by this Apollo::Region.
          num_threads[0] = apollo->ompDefaultNumThreads;
          num_threads[1] = 1;

          num_threads[2] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[3] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[4] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[5] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[6] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[7] = 2;
          num_threads[8] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[9] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[10] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[11] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[12] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[13] = 2;
          num_threads[14] = std::max(2, apollo->numThreadsPerProcCap);
          num_threads[15] = std::min(32, std::max(2, apollo->numThreadsPerProcCap));
          num_threads[16] = std::min(16, std::max(2, (int)(apollo->numThreadsPerProcCap * 0.75)));
          num_threads[17] = std::min(8,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.50)));
          num_threads[18] = std::min(4,  std::max(2, (int)(apollo->numThreadsPerProcCap * 0.25)));
          num_threads[19] = 2;
      }

      const auto l0 = segment_length<Arg0>(data);
      const auto l1 = segment_length<Arg1>(data);
      const auto l2 = segment_length<Arg2>(data);
      auto i0 = l0;
      auto i1 = l1;
      auto i2 = l2;

      using RAJA::internal::thread_privatize;
      auto privatizer = thread_privatize(data);

      apolloRegion->begin();

      apolloRegion->setFeature(l0);
      apolloRegion->setFeature(l1);
      apolloRegion->setFeature(l2);

      policy_index = apolloRegion->getPolicyIndex();

      switch(policy_index) {
          case 0:
              {
                  //std::cout << "3-level OMP defaults" << std::endl; //ggout
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer) \
                  RAJA_COLLAPSE(3)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          for (i2 = 0; i2 < l2; ++i2) {
                              auto& private_data = privatizer.get_priv();
                              private_data.template assign_offset<Arg0>(i0);
                              private_data.template assign_offset<Arg1>(i1);
                              private_data.template assign_offset<Arg2>(i2);
                              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                          }
                      }
                  }
                  break;
              }
          case 1:
              {
                  //std::cout << "Sequential" << std::endl; //ggout
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          for (i2 = 0; i2 < l2; ++i2) {
                              auto& private_data = privatizer.get_priv();
                              private_data.template assign_offset<Arg0>(i0);
                              private_data.template assign_offset<Arg1>(i1);
                              private_data.template assign_offset<Arg2>(i2);
                              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                          }
                      }
                  }
                  break;
              }
          case 2:
          case 3:
          case 4:
          case 5:
          case 6:
          case 7:
              {
                  //std::cout << "OMP static num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(static) \
                  RAJA_COLLAPSE(3)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          for (i2 = 0; i2 < l2; ++i2) {
                              auto& private_data = privatizer.get_priv();
                              private_data.template assign_offset<Arg0>(i0);
                              private_data.template assign_offset<Arg1>(i1);
                              private_data.template assign_offset<Arg2>(i2);
                              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                          }
                      }
                  }
                  break;
              }
          case 8:
          case 9:
          case 10:
          case 11:
          case 12:
          case 13:
              {
                  //std::cout << "OMP dynamic num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(dynamic) \
                  RAJA_COLLAPSE(3)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          for (i2 = 0; i2 < l2; ++i2) {
                              auto& private_data = privatizer.get_priv();
                              private_data.template assign_offset<Arg0>(i0);
                              private_data.template assign_offset<Arg1>(i1);
                              private_data.template assign_offset<Arg2>(i2);
                              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                          }
                      }
                  }
                  break;
              }
          case 14:
          case 15:
          case 16:
          case 17:
          case 18:
          case 19:
              {
                  //std::cout << "OMP guided num_threads:" << num_threads[policy_index] << std::endl; //ggout
#pragma omp parallel for private(i0, i1, i2) firstprivate(privatizer) \
                  num_threads( num_threads[ policy_index ] ) \
                  schedule(guided) \
                  RAJA_COLLAPSE(3)
                  for (i0 = 0; i0 < l0; ++i0) {
                      for (i1 = 0; i1 < l1; ++i1) {
                          for (i2 = 0; i2 < l2; ++i2) {
                              auto& private_data = privatizer.get_priv();
                              private_data.template assign_offset<Arg0>(i0);
                              private_data.template assign_offset<Arg1>(i1);
                              private_data.template assign_offset<Arg2>(i2);
                              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
                          }
                      }
                  }
                  break;
              }


      }

      apolloRegion->end();
  }
};





}  // namespace internal
}  // namespace RAJA

#undef RAJA_COLLAPSE

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
