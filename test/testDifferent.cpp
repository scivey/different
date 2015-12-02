#include "gtest/gtest.h"
#include "different/different.h"
#include <eigen3/Eigen/Dense>
#include <cmath>

namespace different = scivey::different;

double power3(double x) {
  return pow(x, 3);
}

TEST(TestDeriv1, Simple) {
  auto der1 = different::mkDeriv1(power3);
  auto dydx = der1(5.0);
  EXPECT_TRUE(dydx < 76);
  EXPECT_TRUE(dydx > 74);
}

double twoXPower3(double x) {
  return 2 * pow(x, 3);
}

TEST(TestDeriv2, Simple) {
  auto der2 = different::mkDeriv2(twoXPower3);
  auto dydx2 = der2(5.3);
  EXPECT_TRUE(dydx2 < 63.7);
  EXPECT_TRUE(dydx2 > 63.4);
}


double twoXthreeYCubed(Eigen::VectorXd &xy) {
  double x = xy(0);
  double y = xy(1);
  return (2 * pow(x, 3)) + (3 * pow(y, 3));
}

TEST(TestDeriv1Partial, Simple) {
  auto dzdx = different::mkDeriv1(twoXthreeYCubed, 0);
  Eigen::VectorXd args(2);
  args(0) = 5.1;
  args(1) = 4.2;
  auto deriv = dzdx(args);
  EXPECT_TRUE(deriv < 156.1);
  EXPECT_TRUE(deriv > 156.0);

  auto dzdy = different::mkDeriv1(twoXthreeYCubed, 1);
  deriv = dzdy(args);
  EXPECT_TRUE(deriv > 158.7);
  EXPECT_TRUE(deriv < 158.8);
}

TEST(TestDeriv2Partial, Simple) {
  auto dzdx2 = different::mkDeriv2(twoXthreeYCubed, 0);
  Eigen::VectorXd args(2);
  args(0) = 5.1;
  args(1) = 4.2;
  auto deriv = dzdx2(args);
  EXPECT_TRUE(deriv < 61.25);
  EXPECT_TRUE(deriv > 61.15);

  auto dzdy = different::mkDeriv2(twoXthreeYCubed, 1);
  deriv = dzdy(args);
  EXPECT_TRUE(deriv > 75.55);
  EXPECT_TRUE(deriv < 75.65);
}

TEST(TestGradient, Simple) {
  auto gradFn = different::mkGradient(twoXthreeYCubed);
  Eigen::VectorXd grad(2);
  Eigen::VectorXd args(2);
  args(0) = 5.1;
  args(1) = 4.2;
  gradFn(args, grad);

  double dzdx = grad(0);
  EXPECT_TRUE(dzdx < 156.1);
  EXPECT_TRUE(dzdx > 156.0);

  double dzdy = grad(1);
  EXPECT_TRUE(dzdy < 158.8);
  EXPECT_TRUE(dzdy > 158.7);

}

TEST(TestHessian, Simple) {
  auto hessFn = different::mkHessian(twoXthreeYCubed);
  Eigen::MatrixXd hess(2, 2);
  Eigen::VectorXd args(2);
  args(0) = 5.1;
  args(1) = 4.2;
  hessFn(args, hess);

  double dzdx2 = hess(0, 0);
  EXPECT_TRUE(dzdx2 < 61.25);
  EXPECT_TRUE(dzdx2 > 61.15);

  double dzdy2 = hess(1, 1);
  EXPECT_TRUE(dzdy2 < 75.65);
  EXPECT_TRUE(dzdy2 > 75.55);

  double dzdxdy1 = hess(0, 1);
  double dzdxdy2 = hess(1, 0);
  double diff = std::abs(dzdxdy2 - dzdxdy1);
  EXPECT_TRUE(diff < 0.01);

  EXPECT_TRUE(dzdxdy1 > 24776.01);
  EXPECT_TRUE(dzdxdy1 < 24776.15);
}