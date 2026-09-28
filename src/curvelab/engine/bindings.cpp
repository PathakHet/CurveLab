// pybind11 bindings for the incremental curve engine.
//
// The two functions that matter: `on_tick` applies a single update, and the
// parity test uses it to check C++ against NumPy tick by tick. `replay` runs a
// whole sequence of ticks inside C++ and hands back the price and z-score
// matrices, so benchmarks measure the engine and not the Python/C++ call
// overhead.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "curve_engine.hpp"

namespace py = pybind11;
using curvelab::CurveEngine;

PYBIND11_MODULE(_engine, m) {
    m.doc() = "Incremental Brent curve-structure pricing engine (C++)";

    py::class_<CurveEngine>(m, "CurveEngine")
        .def(py::init<std::size_t,
                      const std::vector<std::vector<std::pair<std::uint32_t, double>>>&,
                      std::size_t>(),
             py::arg("n_contracts"), py::arg("legs"), py::arg("window"),
             "Build an engine over `n_contracts` contracts and the given "
             "structure definitions, with a rolling window of `window` updates.")
        .def("on_tick", &CurveEngine::on_tick, py::arg("contract"), py::arg("price"),
             "Apply one contract price update; returns the number of structures touched.")
        .def("price", &CurveEngine::price, py::arg("structure"))
        .def("zscore", &CurveEngine::zscore, py::arg("structure"))
        .def_property_readonly("n_structures", &CurveEngine::n_structures)
        .def_property_readonly("n_contracts", &CurveEngine::n_contracts)
        .def_property_readonly("window", &CurveEngine::window)
        .def_property_readonly("n_updates", &CurveEngine::n_updates)
        .def_property_readonly("fan_out", &CurveEngine::fan_out)
        .def("prices",
             [](const CurveEngine& self) {
                 // Until all of a structure's legs have printed it has no
                 // price, so don't expose the half-built sum.
                 py::array_t<double> out(self.n_structures());
                 double* data = out.mutable_data();
                 for (std::size_t s = 0; s < self.n_structures(); ++s) {
                     data[s] = self.price(static_cast<std::uint32_t>(s));
                 }
                 return out;
             },
             "Snapshot of all structure prices (NaN until every leg has printed).")
        .def("zscores",
             [](const CurveEngine& self) {
                 return py::array_t<double>(self.n_structures(), self.zscores());
             },
             "Snapshot of all structure z-scores.")
        .def("replay",
             [](CurveEngine& self,
                py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast> contracts,
                py::array_t<double, py::array::c_style | py::array::forcecast> prices,
                bool record) -> py::tuple {
                 const auto n = static_cast<std::size_t>(contracts.size());
                 if (static_cast<std::size_t>(prices.size()) != n) {
                     throw std::invalid_argument("contracts and prices must be the same length");
                 }
                 const std::uint32_t* c = contracts.data();
                 const double* p = prices.data();
                 const std::size_t S = self.n_structures();

                 if (!record) {
                     {
                         py::gil_scoped_release release;
                         for (std::size_t i = 0; i < n; ++i) self.on_tick(c[i], p[i]);
                     }
                     return py::make_tuple(py::none(), py::none());
                 }
                 py::array_t<double> out_price({n, S});
                 py::array_t<double> out_z({n, S});
                 double* op = out_price.mutable_data();
                 double* oz = out_z.mutable_data();
                 {
                     py::gil_scoped_release release;
                     for (std::size_t i = 0; i < n; ++i) {
                         self.on_tick(c[i], p[i]);
                         const double* sp = self.prices();
                         const double* sz = self.zscores();
                         for (std::size_t s = 0; s < S; ++s) {
                             op[i * S + s] = self.price(static_cast<std::uint32_t>(s));
                             oz[i * S + s] = sz[s];
                         }
                         (void)sp;
                     }
                 }
                 return py::make_tuple(out_price, out_z);
             },
             py::arg("contracts"), py::arg("prices"), py::arg("record") = true,
             "Replay a tick sequence entirely in C++.");
}
