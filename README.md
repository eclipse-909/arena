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
**Arena**  
An untyped arena allocator that works with the allocator api and Standard types like Box and Vec.
___
## Features
* Simple interface for creating an arena and allocating with it
* Can allocate any Sized data type
* Can allocate Box, Vec, and other standard library smart pointers
* All allocated objects are borrow-checked, and won't compile if they live longer than the arena
* drop is called on all smart pointer objects when they are dropped
* Can pass a backing allocator (default is Global)
___
## How to Use
```rs
use arena::{Arena};

fn main() {
    let arena = Arena::new();
    let num1: &mut i32 = arena.alloc(3);
    let num2: Box<i32> = arena.boxed(3);
}
```
___
One use case is to allocate something that implements Copy, and you get a mutable reference to that thing.
```rs
use arena::Arena;

fn main() {
    let arena = Arena::new();
    let num: &mut i32 = arena.alloc(5);
    *num = 6;
    // arena is dropped here, so num is freed
}
```
___
You can also use it with Rust's types and data structures in the standard library. However,
some smart pointers are unstable. You need the nightly compiler, and you need the allocator_ext
feature.
```rs
use arena::{Arena};

fn main() {
    let arena = Arena::new();
    let mut vec = arena.vec();
    vec.push(5);
    vec.push(6);
    // arena is dropped here, so vec memory is freed
}
```
Most data types that let you use your own allocator should have a new_in function. You can pass in `arena.inner()` because the Arena struct itself
doesn't implement the Allocator trait, but wraps ArenaAlloc which does.
```rs
use arena::{Arena};

fn main() {
    let arena = Arena::new();
    let mut vec = Vec::new_in(arena.inner());
    vec.push(5);
    vec.push(6);
    // arena is dropped here, so vec memory is freed
}
```

___
## Examples
See tests in lib.rs and /tests.
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