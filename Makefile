.PHONY: build test tidy asan tsan release fmt clean distclean help

help:
	@echo "Targets:"
	@echo "  make build     - configure+build debug preset"
	@echo "  make test      - build debug preset and run tests"
	@echo "  make tidy      - run clang-tidy (static analysis)"
	@echo "  make asan      - build+run tests under ASan/UBSan"
	@echo "  make tsan      - build+run tests under ThreadSanitizer"
	@echo "  make release   - configure+build optimized release preset"
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

fmt:
	clang-format -i src/*.cpp include/realtimedecoder/*.hpp test/*.cpp

clean:
	rm -rf build/debug

distclean:
	rm -rf build/
