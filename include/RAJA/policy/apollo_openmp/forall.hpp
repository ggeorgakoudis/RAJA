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

#ifndef RAJA_forall_apollo_openmp_HPP
#define RAJA_forall_apollo_openmp_HPP


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

#include "RAJA/policy/apollo_openmp/policy.hpp"
#include "RAJA/internal/fault_tolerance.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"


// ----------


namespace RAJA
{
namespace policy
{
namespace apollo_omp
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
RAJA_INLINE resources::EventProxy<resources::Host> forall_impl(
    resources::Host &host_res,
    const apollo_omp_parallel_for_exec &,
    Iterable &&iter,
    Func &&loop_body)
{
    static Apollo         *apollo             = Apollo::instance();
    static Apollo::Region *apolloRegion       = nullptr;
    static int             policy_index       = 0;
    static int max_num_threads = std::min( omp_get_max_threads(), omp_get_thread_limit() );
    if (apolloRegion == nullptr) {
        std::string code_location = apollo->getCallpathOffset();
        apolloRegion = new Apollo::Region(
                1, //num features
                code_location.c_str(), // region uid
                POLICY_COUNT // num policies
                );
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
                    auto body = thread_privatize(loop_body);
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
                auto body = thread_privatize(loop_body);
                RAJA_EXTRACT_BED_IT(iter);
                for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                    body.get_priv()(begin_it[i]);
                }
                break;
            }
        case 2: case 3: case 4: case 5: case 6: case 7:
            {
                //std::cout << "Static num_threads " << ( max_num_threads >> ( policy_index - 2) ) << std::endl;
                #pragma omp parallel num_threads( max_num_threads >> ( policy_index - 2) )
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for schedule(static)
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
                    }
                }
                break;
            }
        case 8: case 9: case 10: case 11: case 12: case 13:
            {
                //std::cout << "Dynamic num_threads " << ( max_num_threads >> ( policy_index - 8) ) << std::endl;
                #pragma omp parallel num_threads( max_num_threads >> ( policy_index - 8 ))
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);
                    RAJA_EXTRACT_BED_IT(iter);
                    #pragma omp for schedule(dynamic)
                    for (decltype(distance_it) i = 0; i < distance_it; ++i) {
                        body.get_priv()(begin_it[i]);
                    }
                }
                break;
            }
        case 14: case 15: case 16: case 17: case 18: case 19:
            {
                //std::cout << "Guided num_threads " << ( max_num_threads >> ( policy_index - 14) ) << std::endl;
                #pragma omp parallel num_threads( max_num_threads >> ( policy_index - 14 ) )
                {
                    using RAJA::internal::thread_privatize;
                    auto body = thread_privatize(loop_body);
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

    return resources::EventProxy<resources::Host>(&host_res);
}

//////////
}  // closing brace for apollo namespace
}  // closing brace for policy namespace
}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
