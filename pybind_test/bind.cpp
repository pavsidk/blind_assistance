#include <pybind11/pybind11.h>

namespace py = pybind11;

// forward declare the C++ function
int compute_product(int a, int b);

PYBIND11_MODULE(mybindings, m) {
    m.doc() = "pybind11 test module exposing a simple multiplication function";

    m.def("compute_product", &compute_product,
          "Return the product of two integers a and b");
}
