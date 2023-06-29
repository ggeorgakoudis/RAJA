#ifndef RAJA_policy_apollo_multi_kernel_impl_HPP
#define RAJA_policy_apollo_multi_kernel_impl_HPP

#include "RAJA/config.hpp"
#include "RAJA/policy/apollo_multi/kernel.hpp"
//#include "RAJA/policy/cuda/kernel/CudaKernel.hpp"

#include "apollo/Apollo.h"
#include "apollo/Region.h"

namespace RAJA
{

namespace policy
{

namespace apollo_multi
{

template<typename KernelPolicy>
struct PreLaunchKernel {
  static void exec(Apollo::Region *, Apollo::RegionContext *)
  {}
};

template<typename KernelPolicy>
struct PostLaunchKernel {
  static void exec(Apollo::Region *region, Apollo::RegionContext *context)
  {
    region->end(context);
  }
};

#if defined(RAJA_ENABLE_CUDA)
template<typename... Args>
struct PreLaunchKernel<RAJA::statement::CudaKernelAsync<Args...>> {
  static void exec(Apollo::Region */*region*/, Apollo::RegionContext *context)
  {
    context->timer = Apollo::Timer::create<Apollo::Timer::CudaAsync>();
    context->timer->start();
  }
};
#endif

#if defined(RAJA_ENABLE_HIP)
template<typename... Args>
struct PreLaunchKernel<RAJA::statement::HipKernelAsync<Args...>> {
  static void exec(Apollo::Region */*region*/, Apollo::RegionContext *context)
  {
    context->timer = Apollo::Timer::create<Apollo::Timer::HipAsync>();
    context->timer->start();
  }
};
#endif

template <camp::idx_t idx,
          camp::idx_t num_policies,
          typename KernelPolicyList,
          typename SegmentTuple,
          typename... Bodies>
struct KernelPolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   SegmentTuple &&segments,
                                   Bodies &&... bodies)
  {
    // Get execution policy.
    using ExecutionPolicy = camp::at_v<KernelPolicyList, idx>;
    // Get first stmt from the statement list to identify CUDA async execution.
    using stmt_t = camp::at_v<ExecutionPolicy, 0>;

    //std::cout << "KernelPolicyGeneratorSingle p:" << policy << " r:" << region->name << " ctx:" << context->idx << "\n";
    if (policy == idx) {
      // generate policy variant, calls top-level kernel pattern.
      PreLaunchKernel<stmt_t>::exec(region, context);

      // TODO: can we return an eventproxy for this specific instantiation?

      RAJA::kernel<ExecutionPolicy>(std::forward<SegmentTuple>(segments),
                           std::forward<Bodies>(bodies)...);

      PostLaunchKernel<stmt_t>::exec(region, context);
    } else
      KernelPolicyGeneratorSingle<idx + 1, num_policies, KernelPolicyList, SegmentTuple, Bodies...>::
          generate(policy,
                   region,
                   context,
                   std::forward<SegmentTuple>(segments),
                   std::forward<Bodies>(bodies)...);
  }
};

template <camp::idx_t num_policies,
          typename KernelPolicyList,
          typename SegmentTuple,
          typename... Bodies>
struct KernelPolicyGeneratorSingle<num_policies,
                             num_policies,
                             KernelPolicyList,
                             SegmentTuple,
                             Bodies...> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   SegmentTuple &&,
                                   Bodies &&...)
  {
  }
};

template <typename KernelPolicyList, typename SegmentTuple, typename... Bodies>
static RAJA_INLINE void KernelPolicyGenerator(int policy,
                                        Apollo::Region *region,
                                        Apollo::RegionContext *context,
                                        SegmentTuple &&segments,
                                        Bodies &&...bodies)
{
  //std::cout << "KernelPolicyGenerator p:" << policy << " r:" << region->name << " ctx:" << context->idx << "\n";
  KernelPolicyGeneratorSingle<0,
                        camp::size<KernelPolicyList>::value,
                        KernelPolicyList,
                        SegmentTuple,
                        Bodies...>::generate(policy,
                                        region,
                                        context,
                                        std::forward<SegmentTuple>(segments),
                                        std::forward<Bodies>(bodies)...);
}


/* TEST STUFF */
template <camp::idx_t idx, camp::idx_t num_tuples, typename SegmentTuple>
struct FeatureGeneratorSingle {
  static void generate(SegmentTuple segments, std::vector<float> &features)
  {
    features.push_back(camp::get<idx>(segments).size());

    FeatureGeneratorSingle<idx + 1, num_tuples, SegmentTuple>::generate(
        segments, features);

    return;
  }
};

template <camp::idx_t num_tuples, typename SegmentTuple>
struct FeatureGeneratorSingle<num_tuples, num_tuples, SegmentTuple> {
  static void generate(SegmentTuple, std::vector<float> &) { return; }
};

template <typename SegmentTuple>
struct FeatureGenerator {
  static void generate(SegmentTuple segments, std::vector<float> &features) {
    FeatureGeneratorSingle<0,
                           camp::tuple_size<SegmentTuple>::value,
                           SegmentTuple>::generate(segments, features);
  }
};

/* END OF TEST STUFF */

template <typename KernelPolicyList, typename SegmentTuple, typename... Bodies>
RAJA_INLINE void kernel_impl(ApolloKernelMultiPolicy<KernelPolicyList> &&,
                             SegmentTuple &&segments,
                             Bodies &&...bodies)
{
  static Apollo *apollo = Apollo::instance();
  static Apollo::Region *apolloRegion = nullptr;
  static int policy_index = 0;

  if (apolloRegion == nullptr) {
    std::string code_location = apollo->getCallpathOffset();
    apolloRegion =
        new Apollo::Region(/* num features */ camp::tuple_size<SegmentTuple>::value,
                           /* region id */ code_location.c_str(),
                           /* num policies */ camp::size<KernelPolicyList>::value
        );
  }

  std::vector<float> features;
  FeatureGenerator<SegmentTuple>::generate(segments, features);

  /*std::cout << "features: [ ";
  for(auto e : features)
    std::cout << e << ", ";
  std::cout << " ]\n";*/
  Apollo::RegionContext *context = apolloRegion->begin(features);

  policy_index = apolloRegion->getPolicyIndex(context);
  //std::cout << "policy_index " << policy_index << std::endl;  // ggout

  KernelPolicyGenerator<KernelPolicyList, SegmentTuple, Bodies...>(policy_index,
                                              apolloRegion,
                                              context,
                                              std::forward<SegmentTuple>(segments),
                                              std::forward<Bodies>(bodies)...);
}

}  // namespace apollo_multi

}  // namespace policy

// Range implementation.
namespace statement {
template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          typename... EnclosedStmts>
struct HipKernelApollo
    : public internal::Statement<hip_exec<0>, EnclosedStmts...> {
};

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          typename... EnclosedStmts>
struct HipKernelApolloRuntime
    : public internal::Statement<hip_exec<0>, EnclosedStmts...> {
};

} // namespace statement

namespace internal

{

template <size_t idx,
          size_t num_policies,
          bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_STEP,
          typename Types,
          typename Data,
          typename... EnclosedStmts>
struct RangePolicyGeneratorSingle {
  static RAJA_INLINE void generate(int policy,
                                   Apollo::Region *region,
                                   Apollo::RegionContext *context,
                                   Data &&data)
  {
    // Use Async to select PreLaunch/PostLaunch.
    // TODO: Fix, not every elegant, using HipKernel/HipKernelAsync types.
    using stmt_t = typename std::conditional<Async,
                                    statement::HipKernelAsync<EnclosedStmts...>,
                                    statement::HipKernel<EnclosedStmts...>>::type;

    if (policy == idx) {
      // generate policy variant, calls top-level kernel pattern.
      policy::apollo_multi::PreLaunchKernel<stmt_t>::exec(region, context);

      constexpr size_t GRID_SIZE = GRID_SIZE_START + idx * GRID_SIZE_STEP;
      constexpr size_t BLOCK_SIZE = BLOCK_SIZE_START + idx * BLOCK_SIZE_STEP;

      using stmt_list_t = StatementList<
          statement::HipKernelExt<hip_launch<false, GRID_SIZE, BLOCK_SIZE>,
                                  EnclosedStmts...>>;
      internal::execute_statement_list<stmt_list_t, Types, Data>(data);

      policy::apollo_multi::PostLaunchKernel<stmt_t>::exec(region, context);
    } else
      RangePolicyGeneratorSingle<idx + 1,
                                 num_policies,
                                 Async,
                                 GRID_SIZE_START,
                                 GRID_SIZE_STEP,
                                 BLOCK_SIZE_START,
                                 BLOCK_SIZE_STEP,
                                 Types,
                                 Data,
                                 EnclosedStmts...>::generate(policy,
                                                             region,
                                                             context,
                                                             std::forward<Data>(
                                                                 data));
  }
};

template <size_t num_policies,
          bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_STEP,
          typename Types,
          typename Data,
          typename... EnclosedStmts>
struct RangePolicyGeneratorSingle<num_policies,
                                  num_policies,
                                  Async,
                                  GRID_SIZE_START,
                                  GRID_SIZE_STEP,
                                  BLOCK_SIZE_START,
                                  BLOCK_SIZE_STEP,
                                  Types,
                                  Data,
                                  EnclosedStmts...> {
  static RAJA_INLINE void generate(int,
                                   Apollo::Region *,
                                   Apollo::RegionContext *,
                                   Data &&)
  {
  }
};

template <size_t num_policies,
          bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_STEP,
          typename Types,
          typename Data,
          typename... EnclosedStmts>
static RAJA_INLINE void RangePolicyGenerator(int policy,
                                             Apollo::Region *region,
                                             Apollo::RegionContext *context,
                                             Data &&data)
{
  RangePolicyGeneratorSingle<0,
                             num_policies,
                             Async,
                             GRID_SIZE_START,
                             GRID_SIZE_STEP,
                             BLOCK_SIZE_START,
                             BLOCK_SIZE_STEP,
                             Types,
                             Data,
                             EnclosedStmts...>::generate(policy,
                                                         region,
                                                         context,
                                                         std::forward<Data>(
                                                             data));
}

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<statement::HipKernelApollo<Async,
                                                    GRID_SIZE_START,
                                                    GRID_SIZE_END,
                                                    GRID_SIZE_STEP,
                                                    BLOCK_SIZE_START,
                                                    BLOCK_SIZE_END,
                                                    BLOCK_SIZE_STEP,
                                                    EnclosedStmts...>,
                         Types> {

  template <typename Data>
  static inline void exec(Data &&data)
  {
    static Apollo *apollo = Apollo::instance();
    static Apollo::Region *apolloRegion = nullptr;
    static int policy_index = 0;

    std::vector<float> features;
    policy::apollo_multi::FeatureGenerator<typename camp::decay<
        Data>::segment_tuple_t>::generate(data.segment_tuple, features);

    constexpr size_t num_of_grid_sizes = 1 + (GRID_SIZE_END - GRID_SIZE_START) / GRID_SIZE_STEP;
    constexpr size_t num_of_block_sizes = 1 + (BLOCK_SIZE_END - BLOCK_SIZE_START) / BLOCK_SIZE_STEP;
    constexpr size_t num_policies = num_of_grid_sizes * num_of_block_sizes;

    if (apolloRegion == nullptr) {
      std::string code_location = apollo->getCallpathOffset();
      apolloRegion = new Apollo::Region(
          /* num features */ camp::tuple_size<
              typename camp::decay<Data>::segment_tuple_t>::value,
          /* region id */ code_location.c_str(),
          /* num policies */ num_policies);
    }

    Apollo::RegionContext *context = apolloRegion->begin(features);

    policy_index = apolloRegion->getPolicyIndex(context);

    RangePolicyGenerator<num_policies,
                         Async,
                         GRID_SIZE_START,
                         GRID_SIZE_STEP,
                         BLOCK_SIZE_START,
                         BLOCK_SIZE_STEP,
                         Types,
                         Data,
                         EnclosedStmts...>(policy_index,
                                           apolloRegion,
                                           context,
                                           std::forward<Data>(data));
  }
};

template <bool Async,
          size_t GRID_SIZE_START,
          size_t GRID_SIZE_END,
          size_t GRID_SIZE_STEP,
          size_t BLOCK_SIZE_START,
          size_t BLOCK_SIZE_END,
          size_t BLOCK_SIZE_STEP,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<statement::HipKernelApolloRuntime<Async,
                                                           GRID_SIZE_START,
                                                           GRID_SIZE_END,
                                                           GRID_SIZE_STEP,
                                                           BLOCK_SIZE_START,
                                                           BLOCK_SIZE_END,
                                                           BLOCK_SIZE_STEP,
                                                           EnclosedStmts...>,
                         Types> {
  template <typename Data>
  static inline void exec(Data &&data)
  {
    auto equal_dims = [](auto rhs, const auto lhs) {
      if (rhs.x == lhs.x && rhs.y == lhs.y && rhs.z == lhs.z) return true;
      return false;
    };

    //auto print_dims = [](const char *msg, const auto dims) {
    //  std::cout << msg << " " << dims.x << ", " << dims.y << ", " << dims.z
    //            << "\n";
    //};

    RAJA::resources::Hip res = data.get_resource();
    using stmt_list_t = StatementList<EnclosedStmts...>;
    using data_t = camp::decay<Data>;
    using executor_t = hip_statement_list_executor_t<stmt_list_t, data_t, Types>;
    LaunchDims launch_dims = executor_t::calculateDimensions(data);

    static Apollo *apollo = Apollo::instance();
    static Apollo::Region *apolloRegion = nullptr;
    static int policy_index = 0;

    static size_t num_of_grid_sizes = 1 + (GRID_SIZE_END - GRID_SIZE_START) / GRID_SIZE_STEP;
    static size_t num_of_block_sizes = 1 + (BLOCK_SIZE_END - BLOCK_SIZE_START) / BLOCK_SIZE_STEP;

    if (apolloRegion == nullptr) {
      if (equal_dims(launch_dims.blocks, launch_dims.min_blocks)) {
        //std::cout << "Warning: cannot change blocks\n";
        num_of_grid_sizes = 1;
      }

      if (equal_dims(launch_dims.threads, launch_dims.min_threads)) {
        //std::cout << "Warning: cannot change threads\n";
        num_of_block_sizes = 1;
      }

      static size_t num_policies = num_of_grid_sizes * num_of_block_sizes;
      //std::cout << "num_policies " << num_policies << "\n";

      std::string code_location = apollo->getCallpathOffset();
      apolloRegion = new Apollo::Region(
          /* num features */ camp::tuple_size<
              typename camp::decay<Data>::segment_tuple_t>::value,
          /* region id */ code_location.c_str(),
          /* num policies */ num_policies);
    }

    std::vector<float> features;
    policy::apollo_multi::FeatureGenerator<typename camp::decay<
        Data>::segment_tuple_t>::generate(data.segment_tuple, features);

    Apollo::RegionContext *context = apolloRegion->begin(features);

    policy_index = apolloRegion->getPolicyIndex(context);

    // Iterate policies.
    int policy_index_grid_size = policy_index / num_of_block_sizes;
    int policy_index_block_size = policy_index % num_of_block_sizes;

    size_t GridSize = GRID_SIZE_START + policy_index_grid_size * GRID_SIZE_STEP;
    size_t BlockSize =
        BLOCK_SIZE_START + policy_index_block_size * BLOCK_SIZE_STEP;

    //std::cout << "GridSize " << GridSize << " BlockSize " << BlockSize << "\n";
    launch_dims.blocks = fitHipDims(GridSize, launch_dims.blocks, launch_dims.min_blocks);
    launch_dims.threads = fitHipDims(BlockSize, launch_dims.threads, launch_dims.min_threads);
    //print_dims("effective blocks", launch_dims.blocks);
    //print_dims("effective threads", launch_dims.threads);

    auto hip_data = RAJA::hip::make_launch_body(
        launch_dims.blocks, launch_dims.threads, /*shmem*/ 0, res, data);

    // Use 0 for BlockSize since it is a runtime-value, launch bounds cannot be
    // set.
    using kernelGetter_t = HipKernelLauncherGetter<0, data_t, executor_t>;

    auto func = kernelGetter_t::get();
    static constexpr bool async = Async;
    void *args[] = {(void*)&hip_data};

    // Use Async to select PreLaunch/PostLaunch.
    // TODO: Fix, not every elegant, using HipKernel/HipKernelAsync types.
    using tmp_stmt_t =
        typename std::conditional<Async,
                                  statement::HipKernelAsync<EnclosedStmts...>,
                                  statement::HipKernel<EnclosedStmts...>>::type;
    // TODO: clean up namespaces.
    policy::apollo_multi::PreLaunchKernel<tmp_stmt_t>::exec(apolloRegion, context);

    RAJA::hip::launch((const void *)func,
                      launch_dims.blocks,
                      launch_dims.threads,
                      args,
                      /*shmem*/ 0,
                      res,
                      async);

    policy::apollo_multi::PostLaunchKernel<tmp_stmt_t>::exec(apolloRegion, context);
  }
};

} // namespace internal

}  // namespace RAJA

#endif
