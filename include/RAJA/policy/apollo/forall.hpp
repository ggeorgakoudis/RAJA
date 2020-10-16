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

#ifndef RAJA_forall_apollo_HPP
#define RAJA_forall_apollo_HPP

#include "RAJA/config.hpp"

#include <iostream>
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

#include "RAJA/policy/apollo/policy.hpp"
#include "RAJA/internal/fault_tolerance.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"


// ----------


namespace RAJA
{
namespace policy
{
namespace apollo
{
//
//////////////////////////////////////////////////////////////////////
//
// The following function template switches between various RAJA
// execution policies based on feedback from the Apollo system.
//
//////////////////////////////////////////////////////////////////////
//

#ifndef RAJA_ENABLE_OPENMP
#error "*** RAJA_ENABLE_OPENMP is not defined!" \
    "This build of RAJA requires OpenMP to be enabled! ***"
#endif

template <typename Iterable, typename Func>
RAJA_INLINE void forall_impl(const apollo_exec &, Iterable &&iter, Func &&loop_body)
{
    static Apollo         *apollo             = Apollo::instance();
    static Apollo::Region *apolloRegion       = nullptr;
    static int             policy_index       = 0;
    static int             num_threads[POLICY_COUNT]   = { 0 };
    if (apolloRegion == nullptr) {
        std::string code_location = apollo->getCallpathOffset();
        apolloRegion = new Apollo::Region(
                1, //num features
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

    // Count the number of elements.
    float num_elements = 0.0;
    num_elements = (float) std::distance(std::begin(iter), std::end(iter));

    apolloRegion->begin();
    apolloRegion->setFeature(num_elements);

    policy_index = apolloRegion->getPolicyIndex();
    //std::cout << "policy_index " << policy_index << std::endl; //ggout

    switch(policy_index) {
        case 0:
            {
                //std::cout << "OMP defaults" << std::endl; //ggout
                #pragma omp parallel
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);//.get_priv();
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
                    }
                }
                break;
            }
        case 1:
            {
                //std::cout << "Sequential" << std::endl;
                using RAJA::internal::thread_privatize;
                auto body = thread_privatize(loop_body);//.get_priv();
                RAJA_EXTRACT_BED_IT(iter);
                for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                    body.get_priv()(begin_it[i]);
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
                //std::cout << "Static num_threads " << num_threads[ policy_index ] << std::endl;
                #pragma omp parallel num_threads( num_threads[ policy_index ] )
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);//.get_priv();
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for schedule(static)
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
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
                //std::cout << "Dynamic num_threads " << num_threads[ policy_index ] << std::endl;
                #pragma omp parallel num_threads( num_threads[ policy_index ] )
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);//.get_priv();
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for schedule(dynamic)
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
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
                //std::cout << "Guided num_threads " << num_threads[ policy_index ] << std::endl;
                #pragma omp parallel num_threads( num_threads[ policy_index ] )
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);//.get_priv();
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for schedule(guided)
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
                    }
                }
                break;
            }
    }

    apolloRegion->end();
}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
