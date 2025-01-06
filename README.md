# Arena Allocator
___
## The Problem
I was looking for an arena crate on crates.io, and I was surprised that the most popular ones have
some tradeoffs that I didn't want.

* typed-arena uses typed-arenas (who could've guessed)
* bumpalo doesn't implement the unstable Allocator trait, so you can't use it with types and data
structures in Rust's standard library, and you have to use bumpalo's stand-in types
* slotmap isn't really an arena since you can deallocate individual objects
* The Rust compiler has typed-arenas hard coded into it, but they aren't available in the
standard library (which is the dumbest thing I've ever heard since Rust tries to claim it's a
low-level, systems language)

I'm sure I could have found something I liked, but I also wanted to learn how to make my own to
be a better software developer.
___
## The Solution
I made two implementations of an arena:

**SingleThreadArena** is an Arena that can only be used in a single-threaded environment.  
**MultiThreadArena** is an Arena that can be used in a multithreaded environment.
___
## Features
* Simple interface for creating an arena and allocating with it
* Can allocate any Sized data type
* Can allocate Box, Vec, and other standard library smart pointers (with nightly compiler and
```allocator_api``` feature)
* All allocated objects are borrow-checked, and won't compile if they live longer than the arena
* drop is called on all allocated objects when the arena is dropped
* MultiThreadArena itself can be shared between threads
* Allocated objects can be shared between threads (only if allocated with MultiThreadArena, and if
the arena has a ```'static``` lifetime)
* MultiThreadArena is locked by Mutex when allocations are made, but allocated objects can be
dereferenced at any time without a Mutex (unless the object itself is shared between threads)
___
## How to Use
The Arena struct uses generics to break these two down into two different types, so you can
specify which one you want at compile-time. They work exactly the same, except you can wrap
MultiThreadArena in std::sync::Arc to share it between threads.
```Rust
use arena::{SingleThreadArena, MultiThreadArena};
use std::sync::Arc;

fn main() {
    let st_arena = SingleThreadArena::new();
    let mt_arena = Arc::new(MultiThreadArena::new());
}
```
___
The basic use case is to allocate something, and you get a mutable reference to that thing whose
lifetime is that of the arena.
```Rust
use arena::SingleThreadArena;

fn main() {
    let arena = SingleThreadArena::new();
    let num: &mut i32 = arena.alloc(5);
    *num = 6;
    // arena is dropped here, so num is freed
}
```
___
You can also use it with Rust's types and data structures in the standard library. However,
these features are unstable. You need the nightly compiler, and you need the allocator_api
feature.
```Rust
#![feature(allocator_api)]
use arena::{SingleThreadArena, SingleThreadArenaAlloc};

fn main() {
    let arena = SingleThreadArena::new();
    let mut vec: Vec<i32, &SingleThreadArenaAlloc> = Vec::new_in(arena.allocator());
    vec.push(5);
    vec.push(6);
    // arena is dropped here, so vec is freed
}
```
Most data types that let you use your own allocator should have a new_in function. You'll notice
that we pass in ```arena.allocator()```, which must be done because the Arena struct itself
doesn't implement the Allocator trait, but wraps ArenaAlloc which does.

If you want to specify the type for vec, you must provide a second generic argument which will
be ```&SingleThreadArenaAlloc``` or ```&MultiThreadArenaAlloc```.
___
## Examples
See the test functions at the bottom of lib.rs.
___
## Considerations
This library uses unsafe code. I was pretty careful to make sure there were no memory-related
problems, but I'm not a professional Rust developer, and I'm not too familiar with Rust's
standards and requirements for unsafe code. You can't write unsafe Rust the same way you would
write C or C++. The test functions were run with an address-sanitizer and Miri, and no problems
were found. However, there's still a chance that there is undefined behavior, so use at your own
risk.

If you want to make any contributions, you're free to do so, but I may never get around to
looking at it.