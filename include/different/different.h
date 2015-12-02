#pragma once
#include <functional>
#include <eigen3/Eigen/Dense>
#include <unordered_map>
#include <glog/logging.h>

namespace scivey { namespace different {

enum class ApproxType {
  FORWARD, BACKWARD, CENTRAL
};

const double H_STEP = 0.00001;

double evalSecondDeriv(std::function<double (double)> fn, double x) {
  return (fn(x + H_STEP) - (2* fn(x)) + fn(x - H_STEP)) / pow(H_STEP, 2);
}

double evalSecondDeriv(std::function<double (Eigen::VectorXd&)> fn, size_t idx, Eigen::VectorXd &args) {
  double original = args(idx);
  auto fx = fn(args);
  args(idx) = original - H_STEP;
  auto fxMinusH = fn(args);
  args(idx) = original + H_STEP;
  auto fxPlusH = fn(args);
  args(idx) = original;
  return (fxPlusH - (2 * fx) + fxMinusH) / pow(H_STEP, 2);
}

double evalFirstDerivForward(std::function<double (double)> fn, double x) {
  return (fn(x + H_STEP) - fn(x)) / H_STEP;
}

double evalFirstDerivForward(std::function<double (Eigen::VectorXd&)> fn, size_t idx, Eigen::VectorXd &args) {
  double original = args(idx);
  auto fx1 = fn(args);
  args(idx) = original + H_STEP;
  auto fx2 = fn(args);
  args(idx) = original;
  return (fx2 - fx1) / H_STEP;
}

double evalFirstDerivBackward(std::function<double (double)> fn, double x) {
  return (fn(x) - fn(x - H_STEP)) / H_STEP;
}

double evalFirstDerivBackward(std::function<double (Eigen::VectorXd&)> fn, size_t idx, Eigen::VectorXd &args) {
  double original = args(idx);
  auto fx2 = fn(args);
  args(idx) = original - H_STEP;
  auto fx1 = fn(args);
  args(idx) = original;
  return (fx2 - fx1) / H_STEP;
}

double evalFirstDerivCentral(std::function<double (double)> fn, double x) {
  return (fn(x + H_STEP) - fn(x - H_STEP)) / (H_STEP * 2);
}

double evalFirstDerivCentral(std::function<double (Eigen::VectorXd&)> fn, size_t idx, Eigen::VectorXd &args) {
  double original = args(idx);
  args(idx) = original + H_STEP;
  auto fx2 = fn(args);
  args(idx) = original - H_STEP;
  auto fx1 = fn(args);
  args(idx) = original;
  return (fx2 - fx1) / (H_STEP * 2);
}

double evalFirstDeriv(std::function<double(double)> fn, double x, ApproxType approxType) {
  switch(approxType) {
    case ApproxType::CENTRAL: return evalFirstDerivCentral(fn, x);
    case ApproxType::FORWARD: return evalFirstDerivForward(fn, x);
    case ApproxType::BACKWARD: return evalFirstDerivBackward(fn, x);
  }
}

double evalFirstDeriv(std::function<double (Eigen::VectorXd&)> fn, size_t idx, Eigen::VectorXd &args, ApproxType approxType = ApproxType::CENTRAL) {
  switch(approxType) {
    case ApproxType::CENTRAL: return evalFirstDerivCentral(fn, idx, args);
    case ApproxType::FORWARD: return evalFirstDerivForward(fn, idx, args);
    case ApproxType::BACKWARD: return evalFirstDerivBackward(fn, idx, args);
  }
}

void evalGradient(std::function<double (Eigen::VectorXd&)> fn, Eigen::VectorXd &args, Eigen::VectorXd &gradOut) {
  size_t argSize = args.size();
  for (size_t i = 0; i < argSize; i++) {
    gradOut(i) = evalFirstDeriv(fn, i, args);
  }
}

  namespace detail {

    struct IndexPair {
      size_t i;
      size_t j;
      IndexPair(size_t i, size_t j): i(i), j(j){}
      bool operator==(const IndexPair &other) const {
        return i == other.i && j == other.j;
      }
      struct Hasher {
        size_t operator()(const IndexPair &indexPair) const {
          size_t hash1 = std::hash<size_t>()(indexPair.i);
          size_t hash2 = std::hash<size_t>()(indexPair.j);

          // stolen from boost::hash
          hash1 ^= hash2 + 0x9e3779b9 + (hash1 << 6) + (hash1 >> 2);

          return hash1;
        }
      };
    };

  } // detail

void evalHessian(std::function<double (Eigen::VectorXd&)> fn, Eigen::VectorXd &args, Eigen::MatrixXd &hessOut) {
  std::unordered_map<detail::IndexPair, double, detail::IndexPair::Hasher> seen;
  size_t argSize = args.size();
  for (size_t i = 0; i < argSize; i++) {
    for (size_t j = 0; j < argSize; j++) {
      if (i == j) {
        hessOut(i, j) = evalSecondDeriv(fn, i, args);
      } else {
        auto alreadyCalculated = seen.find(detail::IndexPair(i, j));
        if (alreadyCalculated != seen.end()) {
          hessOut(i, j) = alreadyCalculated->second;
        } else {
          double result = evalFirstDerivCentral(fn, i, args) * evalFirstDerivCentral(fn, j, args);
          seen.insert(std::make_pair(detail::IndexPair(j, i), result));
          hessOut(i, j) = result;
        }
      }
    }
  }
}


std::function<double (double)> mkDeriv1(std::function<double(double)> fn, ApproxType approxType = ApproxType::CENTRAL) {
  return [fn, approxType](double x) {
    return evalFirstDeriv(fn, x, approxType);
  };
}

std::function<double (Eigen::VectorXd&)> mkDeriv1(std::function<double (Eigen::VectorXd&)> fn, size_t idx, ApproxType approxType = ApproxType::CENTRAL) {
  return [fn, idx, approxType](Eigen::VectorXd &x) {
    return evalFirstDeriv(fn, idx, x, approxType);
  };
}

std::function<double (double)> mkDeriv2(std::function<double(double)> fn) {
  return [fn](double x) {
    return evalSecondDeriv(fn, x);
  };
};

std::function<double (Eigen::VectorXd&)> mkDeriv2(std::function<double(Eigen::VectorXd&)> fn, size_t idx) {
  return [fn, idx](Eigen::VectorXd &x) {
    return evalSecondDeriv(fn, idx, x);
  };
};

std::function<void (Eigen::VectorXd&, Eigen::VectorXd&)> mkGradient(std::function<double (Eigen::VectorXd&)> fn) {
  return [fn](Eigen::VectorXd &args, Eigen::VectorXd &gradOut) {
    return evalGradient(fn, args, gradOut);
  };
}

std::function<void (Eigen::VectorXd&, Eigen::MatrixXd&)> mkHessian(std::function<double (Eigen::VectorXd&)> fn) {
  return [fn](Eigen::VectorXd &args, Eigen::MatrixXd &hessOut) {
    evalHessian(fn, args, hessOut);
  };
}

}} // scivey::different
