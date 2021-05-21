#include "array/zfpcarray1.h"
#include "array/zfpcarray2.h"
#include "array/zfpcarray3.h"
#include "array/zfpcarray4.h"
#include "array/zfpfactory.h"
using namespace zfp;

extern "C" {
  #include "constants/1dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray1dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 1; }
};

CArray1dTestEnv* const testEnv = new CArray1dTestEnv;

class CArray1dTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray1dTest

#define ZFP_ARRAY_TYPE const_array1d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array1f
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array2d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array2f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array3d

#define UINT uint64
#define SCALAR double
#define DIMS 1

#include "testConstArrayBase.cpp"
#include "testConstArray1Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
