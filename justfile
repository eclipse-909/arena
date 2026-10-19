# list available recipes
default:
	@just --list

# run test suite
tests:
	cargo +nightly test
	cargo +nightly miri test -- --skip test_compile_error
	MIRIFLAGS=-Zmiri-tree-borrows cargo +nightly miri test -- --skip test_compile_error

# run tests that don't compile and output error messages
gen-comp-err-tests:
	TRYBUILD=overwrite cargo test test_compile_error