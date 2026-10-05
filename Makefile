.PHONY: build test tidy asan tsan release bench gross-golden python-ext fmt clean distclean help

help:
	@echo "Targets:"
	@echo "  make build     - configure+build debug preset"
	@echo "  make test      - build debug preset and run tests"
	@echo "  make tidy      - run clang-tidy (static analysis)"
	@echo "  make asan      - build+run tests under ASan/UBSan"
	@echo "  make tsan      - build+run tests under ThreadSanitizer"
	@echo "  make release   - configure+build optimized release preset"
	@echo "  make bench     - build release and run rtd_bench (ARTIFACT=dir SHOTS=dir)"
	@echo "  make gross-golden - bit-exactness on the gross-code goldens (ARTIFACT=dir GOLDEN=dir)"
	@echo "  make python-ext - build the Python extension rtd._native into python/src/rtd"
	@echo "  make fmt       - clang-format all source files in place"
	@echo "  make clean     - remove build/debug only"
	@echo "  make distclean - remove all build/ output"

build:
	cmake --preset debug
	cmake --build --preset debug

test: build
	ctest --preset debug --output-on-failure

tidy:
	cmake --preset tidy
	cmake --build --preset tidy

asan:
	cmake --preset asan
	cmake --build --preset asan
	ctest --preset asan --output-on-failure

tsan:
	cmake --preset tsan
	cmake --build --preset tsan
	ctest --preset tsan --output-on-failure

release:
	cmake --preset release
	cmake --build --preset release

bench: release
	./build/release/src/bench/rtd_bench --artifact $(ARTIFACT) --shots $(SHOTS)

gross-golden: release
	RTD_GROSS_ARTIFACT=$(ARTIFACT) RTD_GROSS_GOLDEN=$(GOLDEN) \
		./build/release/test/rtd_tests --gtest_filter='GrossGolden.*'

python-ext:
	cmake --preset release -B build/release-py -DRTD_ENABLE_PYTHON=ON \
		-DRTD_PYTHON_OUTPUT_DIR=$(CURDIR)/python/src/rtd
	cmake --build build/release-py --target rtd_python

fmt:
	clang-format -i $$(find include src test -name '*.hpp' -o -name '*.cpp')

clean:
	rm -rf build/debug

distclean:
	rm -rf build/
