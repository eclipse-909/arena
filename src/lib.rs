#![feature(
	allocator_ext,
	slice_ptr_get,
)]

use std::{
	alloc::{AllocError, Allocator, Global, Layout},
	cell::{Cell, UnsafeCell},
	ptr::{self, NonNull},
};

const SEG_ALIGN: usize = 16;
const MAX_GROWTH: usize = 1 << 20;
pub const PAGE_SIZE: usize = 4096;

/// Arena allocator which grows dynamically as you allocate more to it.
/// All memory is freed when the Arena is dropped or cleared.
pub struct Arena<'backing, BackingAlloc: Allocator = Global>(ArenaAlloc<'backing, BackingAlloc>);

impl<'backing> Arena<'backing> {
	/// Creates an empty [Arena].
	pub const fn new() -> Self {
		Self(ArenaAlloc::new_in(&Global))
	}
}

impl<'backing, BackingAlloc: Allocator> Arena<'backing, BackingAlloc> {
	/// Creates an empty [Arena].
	pub const fn new_in(backing: &'backing BackingAlloc) -> Self {
		Self(ArenaAlloc::new_in(backing))
	}

	/// Allocates at least enough space for object and moves object into that space.
	/// Returns a reference to object whose lifetime is the same as the arena.
	/// # Panic
	/// Panics if the underlying heap allocation fails (if your system runs out of memory or something similar).
	pub fn alloc<T: Copy>(&self, object: T) -> &T {
		// SAFETY:
		// memory is initialized with object
		unsafe {
			let p = self.0.allocate(Layout::new::<T>()).unwrap().cast();
			p.write(object);
			p.as_ref()
		}
	}

	/// Allocates at least enough space for object and moves object into that space.
	/// Returns a reference to object whose lifetime is the same as the arena.
	/// # Panic
	/// Panics if the underlying heap allocation fails (if your system runs out of memory or something similar).
	#[allow(clippy::mut_from_ref)]
	pub fn alloc_mut<T: Copy>(&self, object: T) -> &mut T {
		// SAFETY:
		// memory is initialized with object
		unsafe {
			let mut p = self.0.allocate(Layout::new::<T>()).unwrap().cast();
			p.write(object);
			p.as_mut()
		}
	}

	/// Returns a reference to uninitialized memory whose lifetime is the same as the arena.
	/// You must initialize the memory before reading it
	/// # Panic
	/// Panics if the underlying heap allocation fails (if your system runs out of memory or something similar).
	pub fn alloc_uninit<T>(&self, count: usize) -> NonNull<T> {
		self.0
			.allocate(Layout::array::<T>(count).unwrap())
			.unwrap()
			.cast()
	}

	/// Allocates at least enough space for object and moves object into that space.
	/// Returns a Box to object whose lifetime is the same as the arena.
	pub fn boxed<T>(&self, object: T) -> Box<T, &ArenaAlloc<'backing, BackingAlloc>> {
		Box::new_in(object, &self.0)
	}

	/// Returns an empty Vec whose lifetime is the same as the arena that contains its elements.
	pub fn vec<T>(&self) -> Vec<T, &ArenaAlloc<'backing, BackingAlloc>> {
		Vec::new_in(&self.0)
	}

	/// Returns an empty Vec with initial capacity whose lifetime is the same as the arena that contains its elements.
	/// Capacity is the number of elements, not the number of bytes.
	pub fn vec_with_capacity<T>(&self, capacity: usize) -> Vec<T, &ArenaAlloc<'backing, BackingAlloc>> {
		Vec::with_capacity_in(capacity, &self.0)
	}

	/// Copies the string slice into the arena and returns the arena-allocated string slice.
	pub fn str_clone_in(&self, s: &str) -> &str {
		// SAFETY:
		// p is initialized via copy_nonoverlapping.
		// s and p do not overlap because str already existed and p was just allocated.
		// from_raw_parts should hopefully work.
		// str is already utf-8 valid therefore p is.
		unsafe {
			let len = s.len();
			let p = self.alloc_uninit(len).as_ptr();
			ptr::copy_nonoverlapping(s.as_ptr(), p, len);
			let s = std::slice::from_raw_parts(p, len);
			str::from_utf8_unchecked(s)
		}
	}

	/// Invalidates all reference to this arena and de-allocates most memory.
	/// The last segment in the arena is reused to start the arena.
	pub fn clear(&mut self) {
		let a = &mut self.0;
		let segments = a.segments.get_mut();
		let Some(tail) = segments.pop() else { return };

		for seg in segments.drain(..) {
			// SAFETY: allocated from `backing` with this layout; `&mut self` means no outstanding references.
			unsafe { a.backing.deallocate(seg.ptr, seg.layout) };
		}

		let base = tail.ptr.as_ptr();
		a.cursor.set(base);
		// SAFETY: one-past-the-end of the allocation
		a.end.set(unsafe { base.add(tail.layout.size()) });
		segments.push(tail); // can't reallocate: the drain kept the Vec's capacity
	}

	pub fn inner(&self) -> &ArenaAlloc<'backing, BackingAlloc> {
		&self.0
	}
}

/// Fat pointer to the heap-allocated memory.
struct ArenaSegment {
	ptr: NonNull<u8>,
	layout: Layout,
}

impl ArenaSegment {
	fn new_in<BackingAlloc: Allocator>(layout: Layout, backing: BackingAlloc) -> Result<Self, AllocError> {
		let ptr = backing.allocate(layout)?.cast::<u8>();
		Ok(Self { ptr, layout })
	}
}

/// Computes the layout of a new segment that can hold a request of `size`/`align`.
/// Grows geometrically from the previous segment and rounds up to whole pages.
fn new_segment_layout(size: usize, align: usize, min_len: usize) -> Result<Layout, AllocError> {
	let len = size.max(min_len).checked_next_multiple_of(PAGE_SIZE).ok_or(AllocError)?;
	Layout::from_size_align(len, SEG_ALIGN.max(align)).map_err(|_| AllocError)
}

/// Aligned dangling pointer for zero-sized allocations.
fn dangling(align: usize) -> NonNull<u8> {
	// SAFETY: align is a nonzero power of two
	unsafe { NonNull::new_unchecked(ptr::without_provenance_mut::<u8>(align)) }
}

/// The allocator used for [Arena]. This struct is wrapped by [Arena] to hide implementation details and unnecessary functions.
pub struct ArenaAlloc<'backing, BackingAlloc: Allocator = Global> {
	/// Every segment ever allocated. Only used to grow and to free in `Drop`.
	/// The bump pointer below is what the hot path uses, so the fast path never touches this Vec.
	segments: UnsafeCell<Vec<ArenaSegment, &'backing BackingAlloc>>,
	/// Next free byte in the tail segment. Null (with `end` null) when there is no segment yet.
	cursor: Cell<*mut u8>,
	/// One past the last byte of the tail segment.
	end: Cell<*mut u8>,
	backing: &'backing BackingAlloc,
}

impl<'backing, BackingAlloc: Allocator> ArenaAlloc<'backing, BackingAlloc> {
	const fn new_in(backing: &'backing BackingAlloc) -> Self {
		Self {
			segments: UnsafeCell::new(Vec::new_in(backing)),
			cursor: Cell::new(ptr::null_mut()),
			end: Cell::new(ptr::null_mut()),
			backing,
		}
	}

	/// Bytes left in the tail segment.
	#[inline]
	fn remaining(&self) -> usize {
		self.end.get().addr() - self.cursor.get().addr()
	}

	/// Fast path: bump the cursor if the request fits in the tail segment.
	/// Padding is computed from the real address, so it is correct for any base alignment.
	#[inline]
	fn try_bump(&self, size: usize, align: usize) -> Option<NonNull<u8>> {
		let cursor = self.cursor.get();
		// Bytes needed to round the address up to `align` (align is a power of two).
		let pad = cursor.addr().wrapping_neg() & (align - 1);
		let total = pad.checked_add(size)?;
		if total > self.remaining() {
			return None;
		}
		// SAFETY:
		// pad + size <= remaining, so both offsets stay within the tail segment.
		// This is raw pointer arithmetic only; no reference to the segment bytes is created.
		unsafe {
			let start = cursor.add(pad);
			self.cursor.set(start.add(size));
			Some(NonNull::new_unchecked(start))
		}
	}

	#[cold]
	fn alloc_slow(&self, size: usize, align: usize) -> Result<NonNull<u8>, AllocError> {
		// SAFETY: same argument as before: the Vec allocates from `backing`, not `self`.
		let segments = unsafe { &mut *self.segments.get() };
		segments.try_reserve(1).map_err(|_| AllocError)?;

		let prev_len = segments.last().map_or(0, |s| s.layout.size());
		let next_len = prev_len.min(MAX_GROWTH).saturating_mul(2).max(PAGE_SIZE);

		// Oversized request, and there's a tail worth keeping: dedicated segment, bump pointer untouched.
		if !segments.is_empty() && size > next_len / 2 {
			let len = size.checked_next_multiple_of(PAGE_SIZE).ok_or(AllocError)?;
			let layout = Layout::from_size_align(len, SEG_ALIGN.max(align)).map_err(|_| AllocError)?;
			let seg = ArenaSegment::new_in(layout, self.backing)?;
			let p = seg.ptr;
			let tail = segments.len() - 1;
			segments.insert(tail, seg); // keeps the bump segment last
			return Ok(p);
		}

		// Regular segment: becomes the new bump region.
		let layout = new_segment_layout(size, align, next_len)?;
		let seg = ArenaSegment::new_in(layout, self.backing)?;
		let base = seg.ptr.as_ptr();
		self.cursor.set(base);
		// SAFETY: one-past-the-end of the allocation
		self.end.set(unsafe { base.add(layout.size()) });
		segments.push(seg);
		self.try_bump(size, align).ok_or(AllocError)
	}

	/// Returns whether `ptr` (with size `size`) is the most recent allocation.
	/// # Safety
	/// `ptr` must point to a live allocation of exactly `size` bytes from this arena.
	#[inline]
	unsafe fn is_last(&self, ptr: NonNull<u8>, size: usize) -> bool {
		size != 0 && ptr::addr_eq(ptr.as_ptr().add(size), self.cursor.get())
	}
}

// SAFETY:
// Returned blocks are disjoint, live until the arena is dropped, and aligned/sized per the layout.
// `Allocator` requires cloned/moved allocators to keep blocks valid; callers get this via `&ArenaAlloc`,
// which never moves the segments' memory.
unsafe impl<'backing, BackingAlloc: Allocator> Allocator for ArenaAlloc<'backing, BackingAlloc> {
	fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		let size = layout.size();
		let align = layout.align();
		if size == 0 {
			return Ok(NonNull::slice_from_raw_parts(dangling(align), 0));
		}
		let p = match self.try_bump(size, align) {
			Some(p) => p,
			None => self.alloc_slow(size, align)?,
		};
		Ok(NonNull::slice_from_raw_parts(p, size))
	}

	/// Individual allocations are never freed. They are released together when the arena is dropped.
	unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

	/// Gives back the same pointer
	/// if this was the last thing allocated
	/// and there's enough space in the arena.
	/// Otherwise, this creates a new allocation and copies the memory over.
	unsafe fn grow(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		let old_size = old_layout.size();
		let new_size = new_layout.size();
		debug_assert!(
			new_size >= old_size,
			"`new_layout.size()` must be greater than or equal to `old_layout.size()`"
		);

		let extra = new_size - old_size;
		let aligned = ptr.as_ptr().addr() & (new_layout.align() - 1) == 0;

		// Extend in place if this was the last allocation, it still satisfies the new alignment,
		// and the tail segment has room for the extra bytes.
		if aligned && self.is_last(ptr, old_size) && extra <= self.remaining() {
			self.cursor.set(self.cursor.get().add(extra));
			return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
		}

		let new_ptr = self.allocate(new_layout)?;
		// SAFETY: the old block is still live (never freed) so it can't overlap the new one.
		ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_size);
		Ok(new_ptr)
	}

	/// Gives back the same pointer and zeros the new memory
	/// if this was the last thing allocated
	/// and there's enough space in the arena.
	/// Otherwise, this creates a new zeroed allocation and copies the memory over.
	unsafe fn grow_zeroed(
		&self,
		ptr: NonNull<u8>,
		old_layout: Layout,
		new_layout: Layout,
	) -> Result<NonNull<[u8]>, AllocError> {
		let old_size = old_layout.size();
		let new_size = new_layout.size();
		debug_assert!(
			new_size >= old_size,
			"`new_layout.size()` must be greater than or equal to `old_layout.size()`"
		);

		let extra = new_size - old_size;
		let aligned = ptr.as_ptr().addr() & (new_layout.align() - 1) == 0;

		if aligned && self.is_last(ptr, old_size) && extra <= self.remaining() {
			// Zero only the new tail, which starts at `old_size` bytes past `ptr`.
			ptr.as_ptr().add(old_size).write_bytes(0, extra);
			self.cursor.set(self.cursor.get().add(extra));
			return Ok(NonNull::slice_from_raw_parts(ptr, new_size));
		}

		let new_ptr = self.allocate_zeroed(new_layout)?;
		ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), old_size);
		Ok(new_ptr)
	}

	/// If this was the last allocation, gives the freed tail back to the arena.
	unsafe fn shrink(&self, ptr: NonNull<u8>, old_layout: Layout, new_layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
		let old_size = old_layout.size();
		let new_size = new_layout.size();
		debug_assert!(new_size <= old_size);

		let aligned = ptr.as_ptr().addr() & (new_layout.align() - 1) == 0;
		if !aligned {
			// Stricter alignment than the block provides: move it.
			let new_ptr = self.allocate(new_layout)?;
			ptr::copy_nonoverlapping(ptr.as_ptr(), new_ptr.as_mut_ptr(), new_size);
			return Ok(new_ptr);
		}

		if self.is_last(ptr, old_size) {
			self.cursor.set(ptr.as_ptr().add(new_size));
			Ok(NonNull::slice_from_raw_parts(ptr, new_size))
		} else {
			Ok(NonNull::slice_from_raw_parts(ptr, old_size))
		}
	}
}

impl<'backing, BackingAlloc: Allocator> Drop for ArenaAlloc<'backing, BackingAlloc> {
	fn drop(&mut self) {
		for seg in self.segments.get_mut().drain(..) {
			// SAFETY: each segment was allocated from `backing` with exactly this layout.
			unsafe { self.backing.deallocate(seg.ptr, seg.layout) };
		}
	}
}

#[cfg(test)]
mod tests {
	use {
		super::*,
		crate::PAGE_SIZE,
	};

	/// `(base address, length)` of every segment, in `Vec` order (the bump segment is last).
	fn segments<A: Allocator>(arena: &Arena<'_, A>) -> Vec<(usize, usize)> {
		// SAFETY: tests are single-threaded and no `&mut` to the Vec is live here.
		let segs = unsafe { &*arena.0.segments.get() };
		segs.iter().map(|s| (s.ptr.as_ptr().addr(), s.layout.size())).collect()
	}

	fn cursor<A: Allocator>(arena: &Arena<'_, A>) -> usize {
		arena.0.cursor.get().addr()
	}

	#[test]
	fn zero_sized_requests_allocate_no_segment() {
		let arena = Arena::new();
		arena.alloc(());
		arena.str_clone_in("");
		let _v = arena.vec_with_capacity::<()>(10);
		assert!(segments(&arena).is_empty());
	}

	#[test]
	fn segments_grow_geometrically_then_cap() {
		let arena = Arena::new();
		// 8 MB of 1000-byte requests, none of which is oversized.
		for _ in 0..8_000 {
			arena.alloc_uninit::<u8>(1000);
		}
		let sizes: Vec<usize> = segments(&arena).iter().map(|s| s.1).collect();
		assert_eq!(sizes[0], PAGE_SIZE);
		for w in sizes.windows(2) {
			assert_eq!(w[1], w[0].min(MAX_GROWTH) * 2, "sizes: {sizes:?}");
		}
		let capped = sizes.iter().filter(|&&s| s == 2 * MAX_GROWTH).count();
		assert!(capped >= 2, "doubling should stop at 2 * MAX_GROWTH: {sizes:?}");
	}

	#[test]
	fn huge_first_allocation_does_not_inflate_next_segment() {
		let arena = Arena::new();
		// Exactly fills its segment (a multiple of the page size), so the next request must overflow.
		let _huge = arena.vec_with_capacity::<u8>(8 * MAX_GROWTH);
		assert_eq!(segments(&arena).len(), 1);
		arena.alloc(1u8);
		let sizes: Vec<usize> = segments(&arena).iter().map(|s| s.1).collect();
		assert_eq!(sizes.len(), 2);
		assert_eq!(sizes[1], 2 * MAX_GROWTH, "next segment must not be sized from the huge one");
	}

	#[test]
	fn oversized_request_gets_dedicated_segment_before_tail() {
		let arena = Arena::new();
		let a: *const u64 = arena.alloc(1u64);
		let mut big = arena.vec_with_capacity::<u8>(1 << 20);
		big.resize(1 << 20, 0xFF);
		let b: *const u64 = arena.alloc(2u64);

		let segs = segments(&arena);
		assert_eq!(segs.len(), 2);
		assert!(segs[0].1 >= 1 << 20, "dedicated segment goes before the tail");
		assert_eq!(segs[1].1, PAGE_SIZE, "bump segment stays last and untouched");

		let (base, len) = segs[1];
		for p in [a, b] {
			assert!((base..base + len).contains(&p.addr()), "small allocations stay in the tail");
		}
		// SAFETY: same segment
		assert_eq!(b, unsafe { a.add(1) });
	}

	#[test]
	fn clear_keeps_tail_and_resets_cursor() {
		let mut arena = Arena::new();
		for _ in 0..100 {
			arena.alloc_uninit::<u8>(1000);
		}
		let before = segments(&arena);
		assert!(before.len() >= 3, "expected several segments: {before:?}");
		let tail = *before.last().unwrap();

		arena.clear();

		assert_eq!(segments(&arena), vec![tail], "only the tail segment should remain");
		assert_eq!(cursor(&arena), tail.0, "cursor should point at the retained segment's base");
		let first: *const u64 = arena.alloc(5u64);
		assert_eq!(first.addr(), tail.0, "first allocation after clear lands at the retained base");
		assert_eq!(segments(&arena).len(), 1, "no new segment should have been needed");
	}

	#[test]
	fn clear_frees_dedicated_segments() {
		let mut arena = Arena::new();
		arena.alloc(1u64);
		{
			let _big = arena.vec_with_capacity::<u8>(1 << 20);
		}
		assert_eq!(segments(&arena).len(), 2);
		arena.clear();
		assert_eq!(segments(&arena).len(), 1);
	}

	#[test]
	fn clear_on_empty_arena_is_noop() {
		let mut arena = Arena::new();
		arena.clear();
		assert!(segments(&arena).is_empty());
		assert_eq!(*arena.alloc(1u32), 1);
		assert_eq!(segments(&arena).len(), 1);
	}
}
