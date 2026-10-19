use arena::Arena;

// A reference into the arena must not outlive the arena.
fn main() {
	let r = {
		let arena = Arena::new();
		arena.alloc(1)
	};
	println!("{r}");
}
