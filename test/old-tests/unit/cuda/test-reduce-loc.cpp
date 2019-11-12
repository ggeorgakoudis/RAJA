//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing tests for RAJA GPU min-loc reductions.
///

#include <cfloat>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

using namespace RAJA;

using UnitIndexSet = RAJA::TypedIndexSet<RAJA::RangeSegment,
                                         RAJA::ListSegment,
                                         RAJA::RangeStrideSegment>;

constexpr const RAJA::Index_type TEST_VEC_LEN = 1024 * 8;

static const int test_repeat = 10;
static const size_t block_size = 256;

// for setting random values in arrays
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_real_distribution<double> dist(-10, 10);
static std::uniform_real_distribution<double> dist2(0, TEST_VEC_LEN - 1);

struct Index {
  RAJA::Index_type idx;
  RAJA_HOST_DEVICE constexpr Index() : idx(-1) {}
  RAJA_HOST_DEVICE constexpr Index(RAJA::Index_type idx) : idx(idx) {}
  RAJA_HOST_DEVICE constexpr bool operator==(const Index& rhs) const { return (idx == rhs.idx); }
};

template <typename T>
struct LocCompare {
   RAJA_HOST_DEVICE constexpr T defaultVal() const { return T(); }
   RAJA_HOST_DEVICE constexpr T getVal(int idx) const { return T(idx); }
};

template <>
struct LocCompare<Index_type> {
   RAJA_HOST_DEVICE constexpr Index_type defaultVal() const { return -1; }
   RAJA_HOST_DEVICE constexpr Index_type getVal(int idx) const { return idx; }
};

template <typename T>
struct LocConvert {
   RAJA_HOST_DEVICE constexpr T get(const T& idx) const { return idx; }
};

template <>
struct LocConvert<Index> {
   RAJA_HOST_DEVICE constexpr Index_type get(const Index& i) const { return i.idx; }
};

static void reset(double* ptr, long length, double def)
{
  for (long i = 0; i < length; ++i) {
    ptr[i] = def;
  }
}

template <typename T>
struct reduce_applier;
template <typename T, typename U, typename Index>
struct reduce_applier<ReduceMinLoc<T, U, Index>> {
  using IndexType = Index;
  static U def() { return DBL_MAX; }
  static U big() { return -500.0; }
  template <bool B>
  static void updatedvalue(U* dvalue,
                           reduce::detail::ValueLoc<U, Index, B>& randval,
                           reduce::detail::ValueLoc<U, Index, B>& dcurrent)
  {
    if (dvalue[LocConvert<IndexType>().get(randval.loc)] > randval.val) {
      dvalue[LocConvert<IndexType>().get(randval.loc)] = randval.val;
      apply(dcurrent, randval);
    }
  }
  RAJA_HOST_DEVICE static void apply(ReduceMinLoc<T, U, Index> const& r,
                                     U const& val,
                                     Index i)
  {
    r.minloc(val, i);
  }
  template <bool B>
  RAJA_HOST_DEVICE static void apply(reduce::detail::ValueLoc<U, Index, B>& l,
                                     reduce::detail::ValueLoc<U, Index, B> const& r)
  {
    l = l > r ? r : l;
  }
  template <bool B>
  static void cmp(ReduceMinLoc<T, U, Index>& l,
                  reduce::detail::ValueLoc<U, Index, B> const& r)
  {
    ASSERT_FLOAT_EQ(r.val, l.get());
    ASSERT_EQ(r.loc, l.getLoc());
  }
};
template <typename T, typename U, typename Index>
struct reduce_applier<ReduceMaxLoc<T, U, Index>> {
  using IndexType = Index;
  static U def() { return -DBL_MAX; }
  static U big() { return 500.0; }
  template <bool B>
  static void updatedvalue(U* dvalue,
                           reduce::detail::ValueLoc<U, Index, B>& randval,
                           reduce::detail::ValueLoc<U, Index, B>& dcurrent)
  {
    if (randval.val > dvalue[LocConvert<IndexType>().get(randval.loc)]) {
      dvalue[LocConvert<IndexType>().get(randval.loc)] = randval.val;
      apply(dcurrent, randval);
    }
  }
  RAJA_HOST_DEVICE static void apply(ReduceMaxLoc<T, U, Index> const& r,
                                     U const& val,
                                     Index i)
  {
    r.maxloc(val, i);
  }
  template <bool B>
  RAJA_HOST_DEVICE static void apply(reduce::detail::ValueLoc<U, Index, B>& l,
                                     reduce::detail::ValueLoc<U, Index, B> const& r)
  {
    l = l > r ? l : r;
  }
  template <bool B>
  static void cmp(ReduceMaxLoc<T, U, Index>& l,
                  reduce::detail::ValueLoc<U, Index, B> const& r)
  {
    ASSERT_FLOAT_EQ(r.val, l.get());
    ASSERT_EQ(r.loc, l.getLoc());
  }
};

template <typename Reducer>
class ReduceCUDA : public ::testing::Test
{
  using applier = reduce_applier<Reducer>;

public:
  static double* dvalue;
  static void SetUpTestCase()
  {
    cudaErrchk(cudaMallocManaged((void**)&dvalue,
                      sizeof(double) * TEST_VEC_LEN,
                      cudaMemAttachGlobal));
    reset(dvalue, TEST_VEC_LEN, applier::def());
  }
  static void TearDownTestCase() { cudaErrchk(cudaFree(dvalue)); }
};

template <typename Reducer>
double* ReduceCUDA<Reducer>::dvalue = nullptr;


TYPED_TEST_CASE_P(ReduceCUDA);

CUDA_TYPED_TEST_P(ReduceCUDA, generic)
{

  using applier = reduce_applier<TypeParam>;
  using IndexType = typename applier::IndexType;
  using reducer = ReduceCUDA<TypeParam>;
  double* dvalue = reducer::dvalue;
  reset(dvalue, TEST_VEC_LEN, applier::def());

  reduce::detail::ValueLoc<double, IndexType> dcurrent(applier::def(), LocCompare<IndexType>().defaultVal());

  for (int tcount = 0; tcount < test_repeat; ++tcount) {


    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);
    TypeParam dmin2(applier::big(), -1);

    int loops = 16;
    for (int k = 0; k < loops; k++) {

      double droll = dist(mt);
      int index = int(dist2(mt));
      reduce::detail::ValueLoc<double, IndexType> randval(droll, LocCompare<IndexType>().getVal(index));
      applier::updatedvalue(dvalue, randval, dcurrent);

      forall<cuda_exec<block_size>>(RAJA::RangeSegment(0, TEST_VEC_LEN),
                                    [=] RAJA_DEVICE(int i) {
                                      applier::apply(dmin0, dvalue[i], IndexType(i));
                                      applier::apply(dmin1, 2 * dvalue[i], IndexType(i));
                                      applier::apply(dmin2, dvalue[i], IndexType(i));
                                    });

      applier::cmp(dmin0, dcurrent);

      ASSERT_FLOAT_EQ(dcurrent.val * 2, dmin1.get());
      ASSERT_EQ(dcurrent.getLoc(), dmin1.getLoc());
      ASSERT_FLOAT_EQ(applier::big(), dmin2.get());
    }
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 2 runs 2 reductions over complete array using an indexset
//        with two range segments to check reduction object state
//        is maintained properly across kernel invocations.
//
CUDA_TYPED_TEST_P(ReduceCUDA, indexset_align)
{

  using applier = reduce_applier<TypeParam>;
  using IndexType = typename applier::IndexType;
  double* dvalue = ReduceCUDA<TypeParam>::dvalue;

  reset(dvalue, TEST_VEC_LEN, applier::def());

  reduce::detail::ValueLoc<double, IndexType> dcurrent(applier::def(), LocCompare<IndexType>().defaultVal());

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    RangeSegment seg0(0, TEST_VEC_LEN / 2);
    RangeSegment seg1(TEST_VEC_LEN / 2, TEST_VEC_LEN);

    UnitIndexSet iset;
    iset.push_back(seg0);
    iset.push_back(seg1);

    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);

    double droll = dist(mt);
    int index = int(dist2(mt));
    reduce::detail::ValueLoc<double, IndexType> randval(droll, LocCompare<IndexType>().getVal(index));
    applier::updatedvalue(dvalue, randval, dcurrent);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
        iset, [=] RAJA_HOST_DEVICE(int i) {
          applier::apply(dmin0, dvalue[i], IndexType(i));
          applier::apply(dmin1, 2 * dvalue[i], IndexType(i));
        });

    ASSERT_FLOAT_EQ(double(dcurrent), double(dmin0));
    ASSERT_FLOAT_EQ(2 * double(dcurrent), double(dmin1));
    ASSERT_EQ(dcurrent.getLoc(), dmin0.getLoc());
    ASSERT_EQ(dcurrent.getLoc(), dmin1.getLoc());
  }
}

////////////////////////////////////////////////////////////////////////////

//
// test 3 runs 2 reductions over disjoint chunks of the array using
//        an indexset with four range segments not aligned with
//        warp boundaries to check that reduction mechanics don't
//        depend on any sort of special indexing.
//
CUDA_TYPED_TEST_P(ReduceCUDA, indexset_noalign)
{

  using applier = reduce_applier<TypeParam>;
  using IndexType = typename applier::IndexType;
  double* dvalue = ReduceCUDA<TypeParam>::dvalue;

  RangeSegment seg0(1, 230);
  RangeSegment seg1(237, 385);
  RangeSegment seg2(860, 1110);
  RangeSegment seg3(2490, 4003);

  UnitIndexSet iset;
  iset.push_back(seg0);
  iset.push_back(seg1);
  iset.push_back(seg2);
  iset.push_back(seg3);

  for (int tcount = 0; tcount < test_repeat; ++tcount) {

    reset(dvalue, TEST_VEC_LEN, applier::def());

    reduce::detail::ValueLoc<double, IndexType> dcurrent(applier::def(), LocCompare<IndexType>().defaultVal());

    TypeParam dmin0(applier::def(), -1);
    TypeParam dmin1(applier::def(), -1);

    // pick an index in one of the segments
    int index = 97;                     // seg 0
    if (tcount % 2 == 0) index = 297;   // seg 1
    if (tcount % 3 == 0) index = 873;   // seg 2
    if (tcount % 4 == 0) index = 3457;  // seg 3

    double droll = dist(mt);
    reduce::detail::ValueLoc<double, IndexType> randval(droll, LocCompare<IndexType>().getVal(index));
    applier::updatedvalue(dvalue, randval, dcurrent);

    forall<ExecPolicy<seq_segit, cuda_exec<block_size>>>(
        iset, [=] RAJA_DEVICE(int i) {
          applier::apply(dmin0, dvalue[i], IndexType(i));
          applier::apply(dmin1, 2 * dvalue[i], IndexType(i));
        });

    ASSERT_FLOAT_EQ(dcurrent.val, double(dmin0));
    ASSERT_FLOAT_EQ(2 * dcurrent.val, double(dmin1));
    ASSERT_EQ(dcurrent.getLoc(), dmin0.getLoc());
    ASSERT_EQ(dcurrent.getLoc(), dmin1.getLoc());
  }
}

REGISTER_TYPED_TEST_CASE_P(ReduceCUDA,
                           generic,
                           indexset_align,
                           indexset_noalign);

using MinLocTypes =
    ::testing::Types<ReduceMinLoc<RAJA::cuda_reduce, double>>;
INSTANTIATE_TYPED_TEST_CASE_P(MinLoc, ReduceCUDA, MinLocTypes);

using MaxLocTypes =
    ::testing::Types<ReduceMaxLoc<RAJA::cuda_reduce, double>>;
INSTANTIATE_TYPED_TEST_CASE_P(MaxLoc, ReduceCUDA, MaxLocTypes);

using MinLocTypesGenericIndex =
    ::testing::Types<ReduceMinLoc<RAJA::cuda_reduce, double, Index>>;
INSTANTIATE_TYPED_TEST_CASE_P(MinLocGenericIndex, ReduceCUDA, MinLocTypesGenericIndex);

using MaxLocTypesGenericIndex =
    ::testing::Types<ReduceMaxLoc<RAJA::cuda_reduce, double, Index>>;
INSTANTIATE_TYPED_TEST_CASE_P(MaxLocGenericIndex, ReduceCUDA, MaxLocTypesGenericIndex);
