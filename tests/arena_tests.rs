#[cfg(test)]
mod arena_tests {
	use {
		arena::Arena,
		std::{
			alloc::{AllocError, Allocator, Global, Layout},
			cell::Cell,
			ptr::NonNull,
		},
	};

	// ---------------------------------------------------------------------------------------------
	// Helpers
	// ---------------------------------------------------------------------------------------------

	#[repr(align(64))]
	#[derive(Clone, Copy)]
	struct Align64(u8);

	#[repr(align(4096))]
	#[derive(Clone, Copy)]
	struct Align4096(u8);

	#[repr(align(64))]
	#[derive(Clone, Copy)]
	struct ZstAlign64;

	struct DropStruct<'a>(&'a mut usize);
	impl<'a> Drop for DropStruct<'a> {
		fn drop(&mut self) {
			*self.0 += 1;
		}
	}

	fn assert_aligned<T>(r: &T) {
		let addr = (r as *const T).addr();
		assert_eq!(addr % align_of::<T>(), 0, "{addr:#x} is not aligned to {}", align_of::<T>());
	}

	/// Backing allocator that counts live allocations so tests can check that everything is freed.
	#[derive(Default)]
	struct Counting {
		allocs: Cell<usize>,
		frees: Cell<usize>,
	}

	impl Counting {
		fn live(&self) -> usize {
			self.allocs.get() - self.frees.get()
		}
	}

	// SAFETY: forwards to Global, only adding counters.
	unsafe impl Allocator for Counting {
		fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
			self.allocs.set(self.allocs.get() + 1);
			Global.allocate(layout)
		}

		unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
			self.frees.set(self.frees.get() + 1);
			Global.deallocate(ptr, layout)
		}
	}

	// ---------------------------------------------------------------------------------------------
	// Existing tests (with fixes)
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_compile_error() {
		// Run `TRYBUILD=overwrite cargo test test_compile_error` once to generate the .stderr files,
		// then review them and check them in.
		trybuild::TestCases::new().compile_fail("tests/comp_err/*.rs");
	}

	#[test]
	fn test_default() {
		let arena = Arena::new();
		let one = arena.alloc(1);
		let two = arena.boxed(2);
		assert_eq!(*one + *two, 3);
	}

	#[test]
	fn test_many_allocations() {
		const ARR1_LEN: usize = 500;
		const ARR2_LEN: usize = 20;
		const ARR3_LEN: usize = 1024;
		let arena = Arena::new();
		let arr1 = arena.alloc_mut([0; ARR1_LEN]);
		let arr2 = arena.alloc_mut([0; ARR2_LEN]);
		let arr3 = arena.alloc_mut([0; ARR3_LEN]);
		for (i, v) in arr1.iter_mut().enumerate() {
			*v = i;
		}
		for (i, v) in arr2.iter_mut().enumerate() {
			*v = i;
		}
		for (i, v) in arr3.iter_mut().enumerate() {
			*v = i;
		}
		for (i, v) in arr1.iter().enumerate() {
			assert_eq!(*v, i);
		}
		for (i, v) in arr2.iter().enumerate() {
			assert_eq!(*v, i)
		}
		for (i, v) in arr3.iter().enumerate() {
			assert_eq!(*v, i)
		}
		drop(arena);
	}

	#[test]
	fn test_vec_in_arena() {
		let arena = Arena::new();
		let mut vec1 = arena.vec();
		let mut vec2 = arena.vec();
		let mut vec3 = arena.vec();
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
			assert_eq!(vec2[i], i);
			assert_eq!(vec3[i], i);
		}
	}

	#[test]
	fn test_vec_pushed_past_segment_boundary() {
		let arena = Arena::new();
		let mut vec = arena.vec();
		for i in 0..4096u64 {
			vec.push(i)
		}
		for (i, v) in vec.iter().enumerate() {
			assert_eq!(*v, i as u64);
		}
	}

	#[test]
	fn test_repr_align_boxed() {
		let arena = Arena::new();
		let num = arena.boxed(Align64(12));
		assert_eq!(num.0, 12);
		assert_aligned(&*num);
	}

	#[test]
	fn test_dropping() {
		let arena = Arena::new();
		let mut drop_count = 0;
		let drop_box = arena.boxed(DropStruct(&mut drop_count));
		drop(drop_box);
		assert_eq!(drop_count, 1);
		/* test scope */ {
			let _drop_box = arena.boxed(DropStruct(&mut drop_count));
		}
		assert_eq!(drop_count, 2);
	}

	#[test]
	fn test_alignment() {
		let arena = Arena::new();
		let bool0: *const _ = arena.alloc(true);
		let bool1: *const _ = arena.alloc(false);
		let int0: *const _ = arena.alloc(4u32);
		let bool2: *const _ = arena.alloc(true);
		// SAFETY:
		// All pointers come from the same segment, and the offsets stay within it.
		unsafe {
			assert_eq!(bool0.add(1), bool1);
			assert_eq!(bool1.add(3), int0.cast());
			// Fixed: int0 is a *const u32, so add(1) moves 4 bytes.
			assert_eq!(int0.add(1).cast(), bool2);
		}
	}

	// ---------------------------------------------------------------------------------------------
	// Alignment
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_address_based_padding() {
		// The segment base is only guaranteed 16-aligned, so a 64-aligned object after a 1-byte object
		// must be padded using the real address, not just the offset.
		let arena = Arena::new();
		for i in 0..8 {
			arena.alloc(1u8);
			let a = arena.alloc(Align64(i));
			assert_aligned(a);
			assert_eq!(a.0, i);
		}
	}

	#[test]
	fn test_alignment_larger_than_segment_default() {
		let arena = Arena::new();
		arena.alloc(1u8);
		let a = arena.alloc(Align4096(3));
		assert_aligned(a);
		assert_eq!(a.0, 3);
		// The arena keeps working afterwards.
		assert_eq!(*arena.alloc(9u8), 9);
	}

	#[test]
	fn test_mixed_alignments() {
		let arena = Arena::new();
		for i in 0..200u8 {
			assert_eq!(*arena.alloc(i), i);
			assert_aligned(arena.alloc(i as u16));
			assert_aligned(arena.alloc(i as u32));
			assert_aligned(arena.alloc(i as u64));
			assert_aligned(arena.alloc(i as u128));
			assert_aligned(arena.alloc(Align64(i)));
		}
	}

	// ---------------------------------------------------------------------------------------------
	// grow / grow_zeroed / shrink
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_in_place_grow() {
		let arena = Arena::new();
		let mut v = arena.vec::<u64>();
		v.push(0);
		let first = v.as_ptr();
		for i in 1..64u64 {
			v.push(i);
		}
		assert!(v.capacity() >= 64);
		assert_eq!(v.as_ptr(), first, "growing the last allocation should extend it in place");
		for (i, x) in v.iter().enumerate() {
			assert_eq!(*x, i as u64);
		}
		// The cursor must have advanced by exactly the new capacity (not old + new).
		let next: *const u64 = arena.alloc(0xDEADu64);
		// SAFETY: same segment
		assert_eq!(next, unsafe { first.add(v.capacity()) });
	}

	#[test]
	fn test_grow_moves_when_not_last() {
		let arena = Arena::new();
		let mut v = arena.vec::<u64>();
		v.push(1);
		let first = v.as_ptr();
		let _other = arena.alloc(0u64);
		// Force a regrow past the initial capacity.
		for i in 2..=16u64 {
			v.push(i);
		}
		assert_ne!(v.as_ptr(), first, "a non-last block can't grow in place");
		for (i, x) in v.iter().enumerate() {
			assert_eq!(*x, i as u64 + 1);
		}
	}

	#[test]
	fn test_grow_zeroed_in_place_zeroes_only_new_tail() {
		let arena = Arena::new();
		let small = Layout::array::<u8>(16).unwrap();
		let big = Layout::array::<u8>(48).unwrap();
		let p = arena.inner().allocate(big).unwrap().cast::<u8>();
		// SAFETY: p is a live 48-byte block and the last allocation.
		unsafe {
			p.as_ptr().write_bytes(0xFF, 48);
			// Shrinking the last allocation gives the tail back, leaving 0xFF bytes behind it.
			let shrunk = arena.inner().shrink(p, big, small).unwrap().cast::<u8>();
			assert_eq!(shrunk, p);
			let grown = arena.inner().grow_zeroed(shrunk, small, big).unwrap().cast::<u8>();
			assert_eq!(grown, p, "should grow in place");
			let bytes = std::slice::from_raw_parts(grown.as_ptr(), 48);
			assert!(bytes[..16].iter().all(|&b| b == 0xFF), "old contents must survive");
			assert!(bytes[16..].iter().all(|&b| b == 0), "new tail must be zeroed");
		}
	}

	#[test]
	fn test_grow_zeroed_moves_when_not_last() {
		let arena = Arena::new();
		let small = Layout::array::<u8>(16).unwrap();
		let big = Layout::array::<u8>(48).unwrap();
		let p = arena.inner().allocate(small).unwrap().cast::<u8>();
		// SAFETY: p is a live 16-byte block.
		unsafe {
			p.as_ptr().write_bytes(0xAB, 16);
			let _other = arena.inner().allocate(Layout::new::<u64>()).unwrap();
			let q = arena.inner().grow_zeroed(p, small, big).unwrap().cast::<u8>();
			assert_ne!(q, p);
			let bytes = std::slice::from_raw_parts(q.as_ptr(), 48);
			assert!(bytes[..16].iter().all(|&b| b == 0xAB));
			assert!(bytes[16..].iter().all(|&b| b == 0));
		}
	}

	#[test]
	fn test_shrink_reclaims_tail() {
		let arena = Arena::new();
		let mut v = arena.vec_with_capacity::<u64>(16);
		v.extend([1, 2, 3]);
		let p = v.as_ptr();
		v.shrink_to_fit();
		assert_eq!(v.as_ptr(), p);
		assert_eq!(v.capacity(), 3);
		// The next allocation starts right after the shrunk block.
		let next: *const u64 = arena.alloc(9u64);
		// SAFETY: same segment
		assert_eq!(next, unsafe { p.add(3) });
		assert_eq!(&v[..], &[1, 2, 3]);
	}

	#[test]
	fn test_shrink_not_last_keeps_space() {
		let arena = Arena::new();
		let mut v = arena.vec_with_capacity::<u64>(16);
		v.extend([1, 2, 3]);
		let p = v.as_ptr();
		let other: *const u64 = arena.alloc(7u64);
		v.shrink_to_fit();
		assert_eq!(v.as_ptr(), p);
		let next: *const u64 = arena.alloc(8u64);
		// SAFETY: same segment
		assert_eq!(next, unsafe { other.add(1) });
		assert_eq!(&v[..], &[1, 2, 3]);
	}

	#[test]
	fn test_interleaved_vecs_across_segments() {
		let arena = Arena::new();
		let mut a = arena.vec();
		let mut b = arena.vec();
		for i in 0..10_000u32 {
			a.push(i);
			b.push(i * 2);
		}
		for i in 0..10_000u32 {
			assert_eq!(a[i as usize], i);
			assert_eq!(b[i as usize], i * 2);
		}
	}

	// ---------------------------------------------------------------------------------------------
	// Zero-sized allocations
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_zero_sized_types() {
		let arena = Arena::new();
		assert_eq!(*arena.alloc(()), ());
		assert_eq!(*arena.boxed(()), ());
		let mut v = arena.vec::<()>();
		for _ in 0..100 {
			v.push(());
		}
		assert_eq!(v.len(), 100);
		assert_aligned(arena.alloc(ZstAlign64));
	}

	#[test]
	fn test_string() {
		let arena = Arena::new();
		assert_eq!(arena.str_clone_in(""), "");
		let s = arena.str_clone_in("héllo, 世界");
		assert_eq!(s, "héllo, 世界");
		// Empty strings in between must not disturb neighbours.
		let a = arena.str_clone_in("abc");
		let _ = arena.str_clone_in("");
		let b = arena.str_clone_in("def");
		assert_eq!((a, b), ("abc", "def"));
	}

	// ---------------------------------------------------------------------------------------------
	// Oversized requests
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_oversized_request_does_not_abandon_tail() {
		let arena = Arena::new();
		let a: *const u64 = arena.alloc(1u64);
		let mut big = arena.vec_with_capacity::<u8>(1 << 20);
		big.resize(1 << 20, 0xFF);
		let b: *const u64 = arena.alloc(2u64);
		// b continues in the original segment right after a.
		// SAFETY: same segment
		assert_eq!(b, unsafe { a.add(1) });
		// SAFETY: both are live arena allocations
		unsafe {
			assert_eq!(*a, 1);
			assert_eq!(*b, 2);
		}
		assert!(big.iter().all(|&x| x == 0xFF));
	}

	// ---------------------------------------------------------------------------------------------
	// clear
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_clear_on_empty_arena() {
		let mut arena = Arena::new();
		arena.clear();
		arena.clear();
		assert_eq!(*arena.alloc(5), 5);
	}

	#[test]
	fn test_clear_reuses_memory() {
		let mut arena = Arena::new();
		/* fill several segments */ {
			let mut v = arena.vec::<u64>();
			for i in 0..10_000 {
				v.push(i);
			}
			assert_eq!(v.iter().sum::<u64>(), 10_000 * 9_999 / 2);
		}
		arena.clear();
		let first: *const u64 = arena.alloc(1u64);
		arena.clear();
		let second: *const u64 = arena.alloc(2u64);
		assert_eq!(first, second, "clear should restart at the retained segment");

		// Fully usable after clear.
		let mut v = arena.vec::<u64>();
		for i in 0..10_000 {
			v.push(i);
		}
		for (i, x) in v.iter().enumerate() {
			assert_eq!(*x, i as u64);
		}
	}

	// ---------------------------------------------------------------------------------------------
	// Backing allocator
	// ---------------------------------------------------------------------------------------------

	#[test]
	fn test_custom_backing_allocator_frees_everything() {
		let backing = Counting::default();
		{
			let arena = Arena::new_in(&backing);
			for i in 0..10_000u64 {
				assert_eq!(*arena.alloc(i), i);
			}
			let mut v = arena.vec();
			for i in 0..10_000u64 {
				v.push(i);
			}
			assert_eq!(v.len(), 10_000);
			assert!(backing.live() > 1, "expected several segments");
		}
		assert_eq!(backing.live(), 0, "all segments must be returned to the backing allocator");
	}

	#[test]
	fn test_clear_returns_memory_to_backing_allocator() {
		let backing = Counting::default();
		let mut arena = Arena::new_in(&backing);
		for _ in 0..100 {
			arena.alloc_uninit::<u8>(1000);
		}
		let before = backing.live();
		arena.clear();
		let after = backing.live();
		assert!(after < before, "clear should free all but one segment ({before} -> {after})");
		assert!(after >= 1);
		drop(arena);
		assert_eq!(backing.live(), 0);
	}
}
