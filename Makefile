PYTHON ?= python
EXT    := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
INC    := $(shell $(PYTHON) -m pybind11 --includes)
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra -I src/curvelab/engine

.PHONY: all engine test lint bench sample run clean

all: engine test

## Build the C++ pricing engine and its Python binding.
engine: src/curvelab/_engine$(EXT)

src/curvelab/_engine$(EXT): src/curvelab/engine/bindings.cpp src/curvelab/engine/curve_engine.hpp
	g++ $(CXXFLAGS) -shared -fPIC $(INC) $< -o $@

bench/bench_engine: bench/bench_engine.cpp src/curvelab/engine/curve_engine.hpp
	g++ $(CXXFLAGS) $< -o $@

## Full test suite, including the C++/NumPy parity test.
test: engine
	$(PYTHON) -m pytest tests/ -q

lint:
	ruff check src tests bench scripts

## Latency percentiles (native) and the throughput comparison against NumPy.
bench: bench/bench_engine engine
	./bench/bench_engine 11 48 2000000
	@echo
	$(PYTHON) bench/bench_vs_numpy.py

## Regenerate the committed synthetic sample dataset.
sample:
	$(PYTHON) scripts/make_sample_data.py

## Run the pipeline on the sample data and write reports/REPORT.md.
run: engine
	$(PYTHON) -m curvelab.cli run

clean:
	rm -f src/curvelab/_engine*.so bench/bench_engine
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
