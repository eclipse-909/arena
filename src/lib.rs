#![feature(allocator_api)]
#![feature(slice_ptr_get)]

use std::{
	alloc::{AllocError, Allocator, Layout, System},
	cell::{UnsafeCell},
	marker::PhantomData,
	ops::DerefMut,
	ptr::{self, NonNull},
	sync::{Mutex, MutexGuard}
};

///Size of OS page on common operating systems.
const PAGE_SIZE: usize = 4096;

///Arena allocator which grows dynamically as you allocate more to it.
///All memory is freed when the Arena is dropped. Drop is called on each item allocated.
///
///Use [SingleThreadArena] and [MultiThreadArena] for convenience.
pub struct Arena<'arena, I: InteriorMut<'arena, Segments>> {
	allocator: ArenaAlloc<'arena, I>,
}

///Arena allocator which grows dynamically as you allocate more to it.
///All memory is freed when the Arena is dropped. Drop is called on each item allocated.
///This is intended for single-threaded use only.
///
///Even if the arena is not shared between threads, but the data allocated in it is,
///you still must use a [MultiThreadArena].
pub type SingleThreadArena<'arena> = Arena<'arena, UnsafeCell<Segments>>;

///Arena allocator which grows dynamically as you allocate more to it.
///All memory is freed when the Arena is dropped. Drop is called on each item allocated.
///You can obtain shared access to the arena if wrapped in [std::sync::Arc].
///
///If the data allocated in the arena is shared between threads via ```Arc<Mutex<&'arena mut T>>``` or something similar,
///the arena's lifetime must be ```'static```. [Arc] requires its allocator to be ```'static```
///because the compiler cannot guarantee that the arena will outlive any threads that use it.
///
///See [SingleThreadArena] for single-threaded use cases.
pub type MultiThreadArena<'arena> = Arena<'arena, Mutex<Segments>>;

impl<'arena, I: InteriorMut<'arena, Segments>> Arena<'arena, I> {
	///Allocates space for object and moves object into that space. Returns a reference to object whose lifetime is the same as the arena.
	///# Panic
	///Panics if the underlying heap allocation fails (if your system runs out of memory or something similar).
	pub fn alloc<T: Sized>(&self, object: T) -> &mut T {
		let ptr: &mut T = unsafe {
			self.allocator
				.allocate(Layout::new::<T>())
				.unwrap()
				.cast::<T>()
				.as_mut()
		};
		*ptr = object;
		ptr
	}
	
	///Returns a reference to the allocator used by the arena. This way you can use the arena for other things, like with [Vec::new_in].
	#[inline(always)]
	pub const fn allocator(&self) -> &ArenaAlloc<'arena, I> {
		&self.allocator
	}
}
impl<'arena> SingleThreadArena<'arena> {
	///Creates an empty [SingleThreadArena] to be used in a single thread.
	#[inline(always)]
	pub const fn new() -> Self {
		Self {
			allocator: ArenaAlloc {
				segments: UnsafeCell::new(Segments {
					head: None,
					tail: None,
					index: 0,
				}),
				_phantom: PhantomData,
			},
		}
	}
}
impl<'arena> MultiThreadArena<'arena> {
	///Creates an empty [MultiThreadArena] which can be sent between threads. Wrap it in [std::sync::Arc] for shared access to the arena.
	#[inline(always)]
	pub const fn new() -> Self {
		Self {
			allocator: ArenaAlloc {
				segments: Mutex::new(Segments {
					head: None,
					tail: None,
					index: 0,
				}),
				_phantom: PhantomData,
			},
		}
	}
}

//Lifetime hell
///This trait is used to make a common interface for structures that allow interior mutability on immutable data.
///It is only implemented for [UnsafeCell] and [Mutex] for single- and multi-threading respectively.
///It shouldn't be necessary to implement this trait, so you can just ignore its existence.
///If you think you are violating the borrow checker at runtime, you can of course implement it for [std::cell::RefCell],
///Then specify a type alias for ```Arena<'arena, RefCell<Segments>>``` and ```ArenaAlloc<'arena, RefCell<Segments>>```.
pub trait InteriorMut<'arena, T: Sized + 'arena> {
	///The type returned from obtaining interior mutability.
	type MutGuard<'guard>
	where
		'arena: 'guard;
	///Constructor with a given object.
	fn new(obj: T) -> Self;
	///Returns [Self::MutGuard] by obtaining interior mutability.
	fn get_guard_mut<'guard>(&'guard self) -> Self::MutGuard<'guard>
	where
		'arena: 'guard;
	///Obtains a mutable reference from [Self::MutGuard] for convenience.
	fn deref_mut<'guard, 'reference>(
		guard: &'reference mut Self::MutGuard<'guard>,
	) -> &'reference mut T
	where
		'arena: 'guard,
		'guard: 'reference;
}
impl<'arena, T: Sized + 'arena> InteriorMut<'arena, T> for UnsafeCell<T> {
	type MutGuard<'guard>
	= *mut T // There's not a lot of guarding going on here
	where
		'arena: 'guard;
	#[inline(always)]
	fn new(obj: T) -> Self {
		UnsafeCell::new(obj)
	}
	#[inline(always)]
	fn get_guard_mut<'guard>(&'guard self) -> Self::MutGuard<'guard>
	where
		'arena: 'guard,
	{
		self.get()
	}
	#[inline(always)]
	fn deref_mut<'guard, 'reference>(guard: &'reference mut Self::MutGuard<'guard>) -> &'reference mut T
	where
		'arena: 'guard,
		'guard: 'reference,
	{
		unsafe { &mut **guard }
	}
}
impl<'arena, T: Sized + 'arena> InteriorMut<'arena, T> for Mutex<T> {
	type MutGuard<'guard>
	= MutexGuard<'guard, T>
	where
		'arena: 'guard;
	#[inline(always)]
	fn new(obj: T) -> Self {
		Mutex::new(obj)
	}
	#[inline(always)]
	fn get_guard_mut<'guard>(&'guard self) -> Self::MutGuard<'guard>
	where
		'arena: 'guard,
	{
		self.lock().unwrap()
	}
	#[inline(always)]
	fn deref_mut<'guard, 'reference>(
		guard: &'reference mut Self::MutGuard<'guard>,
	) -> &'reference mut T
	where
		'arena: 'guard,
		'guard: 'reference,
	{
		guard.deref_mut()
	}
}

///The allocator used for [Arena]. This struct is wrapped by [Arena] to hide implementation details and unnecessary functions.
///You can just ignore its existence.
///
///Use [SingleThreadArenaAlloc] and [MultiThreadArenaAlloc] for convenience.
pub struct ArenaAlloc<'arena, I: InteriorMut<'arena, Segments>> {
	segments: I,
	_phantom: PhantomData<&'arena ()>,
}

///The allocator used by [Arena] in a single-threaded environment. You might need to use this as a generic argument when using things like [Vec::new_in].
///Otherwise, you can ignore its existence.
pub type SingleThreadArenaAlloc<'arena> = ArenaAlloc<'arena, UnsafeCell<Segments>>;

///The allocator used by [Arena] in a multithreaded environment. You might need to use this as a generic argument when using things like [Vec::new_in].
///Otherwise, you can ignore its existence.
pub type MultiThreadArenaAlloc<'arena> = ArenaAlloc<'arena, Mutex<Segments>>;

unsafe impl<'arena, I: InteriorMut<'arena, Segments>> Allocator for ArenaAlloc<'arena, I> {
	///Allocates to the tail [Segment]. If the tail [Segment] cannot fit the [Layout], a new [Segment] is pushed to the linked list,
	///and the allocation is made to that. If the [Layout]'s size is greater than [PAGE_SIZE], the [Segment] will be dynamically sized to be a multiple of [PAGE_SIZE].
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		let mut segments_guard: I::MutGuard<'_> = self.segments.get_guard_mut();
		let segments: &mut Segments = I::deref_mut(&mut segments_guard);
		let layout_size: usize = layout.size();
		//Get fucked, borrow-checker! We're raw-dogging these pointers!
		let tail: NonNull<Segment> = if segments.head.is_none() {
			//first segment in the linked list
			let new_segment_ptr: NonNull<Segment> = Segment::new(layout_size);
			segments.tail = Some(new_segment_ptr);
			segments.head = Some(new_segment_ptr);
			new_segment_ptr
		} else {
			let tail: NonNull<Segment> = segments.tail.unwrap();
			if segments.index + layout_size >= unsafe { tail.read() }.mem_ptr.len() {
				//Create a new segment and make that the tail
				let new_segment_ptr: NonNull<Segment> = Segment::new(layout_size);
				unsafe { (*tail.as_ptr()).next = Some(new_segment_ptr) };
				segments.tail = Some(new_segment_ptr);
				segments.index = 0;
				new_segment_ptr
			} else {
				tail
			}
		};
		//Allocate the memory in the tail segment by adding the size to the index
		let ret_ptr: NonNull<[u8]> = NonNull::slice_from_raw_parts(
			unsafe { tail.read().mem_ptr.as_non_null_ptr().add(segments.index) },
			layout_size,
		);
		segments.index += layout_size;
		Ok(ret_ptr)
	}
	
	///Memory cannot be deallocated individually from an arena. This function does nothing.
	#[inline(always)]
	unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
	
	///Same as [Allocator::grow] but no memory is deallocated here.
	unsafe fn grow(
		&self,
		ptr: NonNull<u8>,
		old_layout: Layout,
		new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		debug_assert!(
			new_layout.size() >= old_layout.size(),
			"`new_layout.size()` must be greater than or equal to `old_layout.size()`"
		);
		let new_ptr: NonNull<[u8]> = self.allocate(new_layout)?;
		
		// SAFETY: because `new_layout.size()` must be greater than or equal to
		// `old_layout.size()`, both the old and new memory allocation are valid for reads and
		// writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
		// deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
		// safe. The safety contract for `dealloc` must be upheld by the caller.
		unsafe {
			ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
			// self.deallocate(ptr, old_layout);
		}
		Ok(new_ptr)
	}
	
	///Same as [Allocator::grow_zeroed] but no memory is deallocated here.
	unsafe fn grow_zeroed(
		&self,
		ptr: NonNull<u8>,
		old_layout: Layout,
		new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		debug_assert!(
			new_layout.size() >= old_layout.size(),
			"`new_layout.size()` must be greater than or equal to `old_layout.size()`"
		);
		let new_ptr: NonNull<[u8]> = self.allocate_zeroed(new_layout)?;
		
		// SAFETY: because `new_layout.size()` must be greater than or equal to
		// `old_layout.size()`, both the old and new memory allocation are valid for reads and
		// writes for `old_layout.size()` bytes. Also, because the old allocation wasn't yet
		// deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
		// safe. The safety contract for `dealloc` must be upheld by the caller.
		unsafe {
			ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_layout.size());
			// self.deallocate(ptr, old_layout);
		}
		Ok(new_ptr)
	}
	
	///Shrinking is not supported in Arenas because individual objects are not deallocated.
	///Instead, this function asserts that the [new_layout]'s size is less than or equal to that of [old_layout].
	///Then it returns [ptr] cast as a slice pointer with the size of [old_layout].
	unsafe fn shrink(
		&self,
		ptr: NonNull<u8>,
		old_layout: Layout,
		new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		debug_assert!(
			new_layout.size() <= old_layout.size(),
			"`new_layout.size()` must be smaller than or equal to `old_layout.size()`"
		);
		
		// let new_ptr = self.allocate(new_layout)?;
		//
		// // SAFETY: because `new_layout.size()` must be lower than or equal to
		// // `old_layout.size()`, both the old and new memory allocation are valid for reads and
		// // writes for `new_layout.size()` bytes. Also, because the old allocation wasn't yet
		// // deallocated, it cannot overlap `new_ptr`. Thus, the call to `copy_nonoverlapping` is
		// // safe. The safety contract for `dealloc` must be upheld by the caller.
		// unsafe {
		// 	ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_layout.size());
		// 	self.deallocate(ptr, old_layout);
		// }
		//
		// Ok(new_ptr)
		
		Ok(NonNull::slice_from_raw_parts(ptr, old_layout.size()))
	}
}
unsafe impl<'arena> Send for ArenaAlloc<'arena, Mutex<Segments>> {}
unsafe impl<'arena> Sync for ArenaAlloc<'arena, Mutex<Segments>> {}

///Linked list of [Segment] nodes used by [ArenaAllocator]. You can ignore its existence.
pub struct Segments {
	///First allocated segment - this is needed to deallocate.
	head: Option<NonNull<Segment>>,
	///Current working segment - this is a convenience so we don't have to traverse the linked list.
	tail: Option<NonNull<Segment>>,
	///Index of the first free byte in the current segment,
	index: usize,
}
impl Drop for Segments {
	fn drop(&mut self) {
		let mut curr_segment: Option<NonNull<Segment>> = self.head;
		while let Some(curr) = curr_segment {
			unsafe {
				curr_segment = curr.read().next;
				//deallocate the memory held by the segment
				System.deallocate(
					curr.read().mem_ptr.cast::<u8>(),
					Layout::from_size_align(curr.read().mem_ptr.len(), 1).unwrap(),
				);
				//deallocate the segment itself
				System.deallocate(curr.cast(), Layout::new::<Segment>());
			}
		}
	}
}

///Linked-list node for [Segments]. You can ignore its existence.
pub struct Segment {
	///Fat pointer to the heap-allocated memory.
	mem_ptr: NonNull<[u8]>,
	///The next Segment in the linked-list.
	next: Option<NonNull<Self>>
}
impl Segment {
	fn new(size: usize) -> NonNull<Self> {
		//allocate the segment itself
		let new_segment_ptr: NonNull<Self> = System
			.allocate(Layout::new::<Self>())
			.unwrap()
			.cast::<Self>();
		unsafe {
			new_segment_ptr.write(Self {
				//allocate memory for the segment
				mem_ptr: System
					.allocate(
						Layout::from_size_align((size / PAGE_SIZE + 1) * PAGE_SIZE, 1).unwrap(),
					)
					.unwrap(),
				next: None,
			})
		};
		new_segment_ptr
	}
}

#[cfg(test)]
mod tests {
	use {
		crate::{
			MultiThreadArena, MultiThreadArenaAlloc, SingleThreadArena, SingleThreadArenaAlloc,
		},
		std::{
			sync::{Arc, RwLock, RwLockWriteGuard},
			thread::{spawn, JoinHandle},
		},
	};
	
	// ///This function shouldn't compile
	// #[test]
	// fn ref_lifetime() {
	// 	let mut arena: SingleThreadArena = SingleThreadArena::new();
	// 	let one: &mut u8 = arena.alloc_move(1);
	// 	drop(arena);
	// 	assert_eq!(*one, 1);//compilation error - borrow checker violated - use after free
	// }
	//
	// ///This function shouldn't compile
	// #[test]
	// fn box_lifetime() {
	// 	let mut arena: SingleThreadArena = SingleThreadArena::new();
	// 	let one: Box<u8, &SingleThreadArenaAlloc> = Box::new_in(1, arena.allocator());
	// 	drop(arena);
	// 	assert_eq!(*one, 1);//compilation error - borrow checker violated - use after free
	// }
	
	#[test]
	fn single_thread() {
		let arena: SingleThreadArena = SingleThreadArena::new();
		let one: &mut u8 = arena.alloc(1);
		let two: Box<u8, &SingleThreadArenaAlloc> = Box::new_in(2, arena.allocator());
		assert_eq!(*one + *two, 3);
	}
	
	#[test]
	fn multi_thread() {
		let arena: Arc<MultiThreadArena> = Arc::new(MultiThreadArena::new());
		let arena_clone: Arc<MultiThreadArena> = arena.clone();
		let handle: JoinHandle<u8> = spawn(move || {
			let one: &mut u8 = arena_clone.alloc(1);
			let two: Box<u8, &MultiThreadArenaAlloc> = Box::new_in(2, arena_clone.allocator());
			*one + *two
		});
		let three: &mut u8 = arena.alloc(3);
		let four: Box<u8, &MultiThreadArenaAlloc> = Box::new_in(4, arena.allocator());
		assert_eq!(handle.join().unwrap() + *three + *four, 10);
	}
	
	#[test]
	fn many_allocations() {
		const ARR1_LEN: usize = 500;
		const ARR2_LEN: usize = 20;
		const ARR3_LEN: usize = 1024;
		let arena: SingleThreadArena = SingleThreadArena::new();
		let arr1: &mut [usize; ARR1_LEN] = arena.alloc([0; ARR1_LEN]);
		let arr2: &mut [usize; ARR2_LEN] = arena.alloc([0; ARR2_LEN]);
		let arr3: &mut [usize; ARR3_LEN] = arena.alloc([0; ARR3_LEN]);
		for i in 0..ARR1_LEN {
			arr1[i] = i;
		}
		for i in 0..ARR2_LEN {
			arr2[i] = i;
		}
		for i in 0..ARR3_LEN {
			arr3[i] = i;
		}
		for i in 0..ARR1_LEN {
			assert_eq!(arr1[i], i);
		}
		for i in 0..ARR2_LEN {
			assert_eq!(arr2[i], i);
		}
		for i in 0..ARR3_LEN {
			assert_eq!(arr3[i], i);
		}
		drop(arena);
	}
	
	#[test]
	fn many_threads() {
		const ARR1_LEN: usize = 500;
		const ARR2_LEN: usize = 20;
		const ARR3_LEN: usize = 1024;
		let arena: Arc<MultiThreadArena> = Arc::new(MultiThreadArena::new());
		let mut threads: Vec<JoinHandle<()>> = Vec::new();
		for _ in 0..12 {
			let arena_clone: Arc<MultiThreadArena> = arena.clone();
			threads.push(spawn(move || {
				let arr1: &mut [usize; ARR1_LEN] = arena_clone.alloc([0; ARR1_LEN]);
				let arr2: &mut [usize; ARR2_LEN] = arena_clone.alloc([0; ARR2_LEN]);
				let arr3: &mut [usize; ARR3_LEN] = arena_clone.alloc([0; ARR3_LEN]);
				for i in 0..ARR1_LEN {
					arr1[i] = i;
				}
				for i in 0..ARR2_LEN {
					arr2[i] = i;
				}
				for i in 0..ARR3_LEN {
					arr3[i] = i;
				}
				for i in 0..ARR1_LEN {
					assert_eq!(arr1[i], i);
				}
				for i in 0..ARR2_LEN {
					assert_eq!(arr2[i], i);
				}
				for i in 0..ARR3_LEN {
					assert_eq!(arr3[i], i);
				}
			}));
		}
		for thread in threads {
			thread.join().unwrap();
		}
	}
	
	#[test]
	fn single_thread_vector() {
		let arena: SingleThreadArena = SingleThreadArena::new();
		let mut vec1: Vec<usize, &SingleThreadArenaAlloc> = Vec::new_in(arena.allocator());
		let mut vec2: Vec<usize, &SingleThreadArenaAlloc> = Vec::new_in(arena.allocator());
		let mut vec3: Vec<usize, &SingleThreadArenaAlloc> = Vec::new_in(arena.allocator());
		for i in 0..32 {
			vec1.push(i);
		}
		for i in 0..32 {
			vec2.push(i);
		}
		for i in 0..32 {
			vec3.push(i);
		}
		vec1.shrink_to_fit();
		vec2.shrink_to_fit();
		vec3.shrink_to_fit();
		for i in 0..32 {
			assert_eq!(vec1[i], i);
		}
		for i in 0..32 {
			assert_eq!(vec2[i], i);
		}
		for i in 0..32 {
			assert_eq!(vec3[i], i);
		}
	}
	
	#[test]
	fn multi_thread_vector() {
		const THREADS: usize = 12;
		
		// The arena must be a MultiThreadArena even though the arena itself is not being shared.
		// Box::into_raw provides a raw pointer, which can be cast into a static immutable reference with as_ref.
		// The arena needs to be static to satisfy the requirement that Arc have a static allocator.
		// The compiler can't detect that we are joining all the threads before dropping the arena,
		// so we have to do all this shit-fuckery to bypass the borrow-checker. We also have to use this scope
		// to tell Miri that the static reference to the arena won't outlive the raw pointer.
		// There are other options, like creating a static variable to store the arena, but making it truly
		// static defeats the purpose of the arena, since the operating system will just free the memory anyway
		// when the process exits. By that point, you might as well leak the memory to save time. There may be a better way to
		// trick the compiler into thinking the arena is static, but this is the best I could come up with.
		let arena_ptr: *mut MultiThreadArena = Box::into_raw(Box::new(MultiThreadArena::new()));
		{
			let arena: &'static MultiThreadArena = unsafe { &*arena_ptr };
			let vec: Arc<RwLock<Vec<usize, &MultiThreadArenaAlloc>>> =
				Arc::new(RwLock::new(Vec::new_in(arena.allocator())));
			let mut threads: Vec<JoinHandle<()>> = Vec::new();
			for t in 0..THREADS {
				let vec_clone: Arc<RwLock<Vec<usize, &MultiThreadArenaAlloc>>> = vec.clone();
				threads.push(spawn(move || {
					vec_clone.write().unwrap().push(t);
				}));
			}
			for thread in threads {
				thread.join().unwrap();
			}
			let mut vec_guard: RwLockWriteGuard<Vec<usize, &MultiThreadArenaAlloc>> =
				vec.write().unwrap();
			vec_guard.sort();
			for t in 0..THREADS {
				assert_eq!(vec_guard[t], t);
			}
		}
		let _ = unsafe { Box::from_raw(arena_ptr) };//Do this if you don't want to leak the memory
	}
	
	struct Vec2 {
		x: f32,
		y: f32,
	}
	impl Drop for Vec2 {
		fn drop(&mut self) {
			println!("Vec2 dropped successfully!"); //This gets printed to the console, so it passes the test
		}
	}
	
	#[test]
	fn dropping() {
		let arena: SingleThreadArena = SingleThreadArena::new();
		{
			println!("Attempting to drop vec_ref..."); //I have to call this up here, probably due to stdout buffering and compiler optimizations
			let vec_ref: &mut Vec2 = arena.alloc(Vec2 { x: 0., y: 0. });
			vec_ref.x = 1.;
			vec_ref.y = 2.;
			//vec_ref dropped here
		}
		{
			println!("Attempting to drop vec_box...");
			let mut vec_box: Box<Vec2, &SingleThreadArenaAlloc> = Box::new_in(Vec2 {x: 0., y: 0.}, arena.allocator());
			vec_box.x = 3.;
			vec_box.y = 4.;
			//vec_box dropped here
		}
		println!("Dropping arena...");
		//arena dropped here
	}
}