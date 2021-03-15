#ifndef RAJA_policy_apollo_multi_kernel_HPP
#define RAJA_policy_apollo_multi_kernel_HPP

#include "RAJA/policy/apollo_multi/kernel/kernel_impl.hpp"

namespace RAJA
{

inline namespace policy_by_value_interface
{

template <typename KernelPolicyList, typename SegmentTuple, typename... Bodies>
RAJA_INLINE void kernel(ApolloMultiKernelPolicy<KernelPolicyList> &p, SegmentTuple &&segments, Bodies &&... bodies) {
  using policy::apollo_multi::kernel_impl;
  kernel_impl(p,
              std::forward<SegmentTuple>(segments),
              std::forward<Bodies>(bodies)...);
}

template <typename PolicyType, typename SegmentTuple, typename... Bodies>
RAJA_INLINE
concepts::enable_if<
    type_traits::is_apollo_multi_policy<PolicyType>
>
kernel(SegmentTuple &&segments, Bodies &&... bodies)
{
    PolicyType p;
    policy_by_value_interface::kernel(p, std::forward<SegmentTuple>(segments), std::forward<Bodies>(bodies)...);
}

}

} // namespace RAJA

#endif  // closing endif for header file include guard
