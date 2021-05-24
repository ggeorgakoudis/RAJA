/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing Apollo wrapper to RAJA scan declarations.
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

#ifndef RAJA_scan_apollo_HPP
#define RAJA_scan_apollo_HPP

#include "RAJA/config.hpp"

#include <algorithm>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <omp.h>


namespace RAJA
{
namespace impl
{
namespace scan
{


/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <typename Policy, typename Iter, typename BinFn>
concepts::enable_if<type_traits::is_apollo_policy<Policy>> inclusive_inplace(
    //const RAJA::apollo_exec&,
    //const RAJA::policy::apollo::apollo_exec,
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f)
{
    RAJA::impl::scan::inclusive_inplace(RAJA::omp_parallel_for_exec{}, begin, end, f);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <typename Policy, typename Iter, typename BinFn, typename ValueT>
concepts::enable_if<type_traits::is_apollo_policy<Policy>> exclusive_inplace(
    //const RAJA::apollo_exec&,
    //const RAJA::policy::apollo::apollo_exec,
    const Policy&,
    Iter begin,
    Iter end,
    BinFn f,
    ValueT v)
{
    RAJA::impl::scan::exclusive_inplace(RAJA::omp_parallel_for_exec{}, begin, end, f, v);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy, typename Iter, typename OutIter, typename BinFn>
concepts::enable_if<type_traits::is_apollo_policy<Policy>> inclusive(
    //const RAJA::apollo_exec&,
    //const RAJA::policy::apollo::apollo_exec,
    const Policy&,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f)
{
    RAJA::impl::scan::inclusive(RAJA::omp_parallel_for_exec{}, begin, end, out, f);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <typename Policy,
          typename Iter,
          typename OutIter,
          typename BinFn,
          typename ValueT>
concepts::enable_if<type_traits::is_apollo_policy<Policy>> exclusive(
    //const RAJA::apollo_exec&,
    //const RAJA::policy::apollo::apollo_exec,
    const Policy&,
    Iter begin,
    Iter end,
    OutIter out,
    BinFn f,
    ValueT v)
{
    RAJA::impl::scan::exclusive(RAJA::omp_parallel_for_exec{}, begin, end, out, f, v);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif
