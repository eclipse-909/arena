use arena::Arena;

// Arena-backed collections borrow the arena too.
fn main() {
	let v = {
		let arena = Arena::new();
		let mut v = arena.vec::<i32>();
		v.push(1);
		v
	};
	println!("{v:?}");
}
