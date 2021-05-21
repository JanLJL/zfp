#include "array/zfpcarray1.h"
#include "array/zfpcarray2.h"
#include "array/zfpcarray3.h"
#include "array/zfpcarray4.h"
#include "array/zfpfactory.h"
using namespace zfp;

extern "C" {
  #include "constants/3dDouble.h"
}

#include "gtest/gtest.h"
#include "utils/gtestDoubleEnv.h"
#include "utils/gtestBaseFixture.h"
#include "utils/predicates.h"

class CArray3dTestEnv : public ArrayDoubleTestEnv {
public:
  virtual int getDims() { return 3; }
};

CArray3dTestEnv* const testEnv = new CArray3dTestEnv;

class CArray3dTest : public CArrayNdTestFixture {};

#define TEST_FIXTURE CArray3dTest

#define ZFP_ARRAY_TYPE const_array3d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR const_array3f
#define ZFP_ARRAY_TYPE_WRONG_DIM const_array4d
#define ZFP_ARRAY_TYPE_WRONG_SCALAR_DIM const_array4f
#define ZFP_ARRAY_NOT_INCLUDED_TYPE const_array2d

#define UINT uint64
#define SCALAR double
#define DIMS 3

#include "testConstArrayBase.cpp"
#include "testConstArray3Base.cpp"

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  static_cast<void>(::testing::AddGlobalTestEnvironment(testEnv));
  return RUN_ALL_TESTS();
}
