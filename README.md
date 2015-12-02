## different

Simple numerical differentiation for C++.

First and second derivatives of single-parameter functions; first and second partial derivatives, gradients and Hessians of functions taking `Eigen::VectorXd` references for arguments.

Supports central, forward and backward approximations for first derivatives.

Library is header-only; compilation is only required to run the tests.

Requires c++11-compatible compiler (`std::function`) and Eigen3.

```c++
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <different/different.h>

using namespace std;
using scivey::different::mkGradient;

double someFunc(Eigen::VectorXd &params) {
    double x = params(0);
    double y = params(1);
    return 2 * pow(x, 3) + 3 * pow(y, 3);
}

int main() {
    Eigen::VectorXd args(2);
    args << 5.1, 4.2;
    Eigen::VectorXd grad(2);
    auto gradFn = mkGradient(someFunc);
    gradFn(args, grad);
    cout << endl << "\t" << grad(0) << "\t" << grad(1) << endl;
    return 0;
}
```

```bash
    156.06    158.76
```


## License

The MIT License (MIT)

Copyright (c) 2015 Scott Ivey

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



