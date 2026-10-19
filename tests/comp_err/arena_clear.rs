use arena::Arena;

// `clear` needs `&mut self`, so it must be rejected while a reference is still live.
fn main() {
	let mut arena = Arena::new();
	let r = arena.alloc(1);
	arena.clear();
	println!("{r}");
}
