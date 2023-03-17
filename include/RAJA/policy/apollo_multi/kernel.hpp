#ifndef RAJA_policy_apollo_multi_kernel_HPP
#define RAJA_policy_apollo_multi_kernel_HPP

#include "RAJA/policy/apollo_multi/kernel/kernel_impl.hpp"

namespace RAJA
{

inline namespace policy_by_value_interface
{

template <typename KernelPolicyList, typename SegmentTuple, typename... Bodies>
RAJA_INLINE void kernel(ApolloKernelMultiPolicy<KernelPolicyList> &&p, SegmentTuple &&segments, Bodies &&... bodies) {
  using policy::apollo_multi::kernel_impl;
  kernel_impl(std::forward<ApolloKernelMultiPolicy<KernelPolicyList>>(p),
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
    policy_by_value_interface::kernel(std::forward<PolicyType>(p),
                                      std::forward<SegmentTuple>(segments),
                                      std::forward<Bodies>(bodies)...);
}

template <typename PolicyType,
          typename SegmentTuple,
          typename Resource,
          typename... Bodies>
RAJA_INLINE
concepts::enable_if_t<
    resources::EventProxy<Resource>,
    type_traits::is_apollo_multi_policy<PolicyType>
>
kernel_resource(SegmentTuple &&segments, Resource resource, Bodies &&...bodies)
{
    PolicyType p;
    policy_by_value_interface::kernel(std::forward<PolicyType>(p),
                                      std::forward<SegmentTuple>(segments),
                                      std::forward<Bodies>(bodies)...);
    return resources::EventProxy<Resource>(resource);
}
}

} // namespace RAJA

#endif  // closing endif for header file include guard
