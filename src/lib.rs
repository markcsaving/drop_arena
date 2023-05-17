#![warn(missing_docs)]

//! This crate provides a custom allocator that can allocate items of a single type, much like
//! [typed-arena](https://docs.rs/crate/typed-arena/latest). A `typed_arena::Arena<T>` allocates
//! a `T` and returns a `&mut T`; the `T` will be dropped when the arena itself goes out of scope.
//!
//! By contrast, a [`DropArena<T>`] allocates a `T` and returns a [`DropBox<T>`]. As the
//! name suggests, a [`DropBox<T>`] is very similar to a [`Box<T>`]. While a [`Box<T>`] is tied to the
//! single global allocator, a [`DropBox<T>`] is tied to the lifetime of the [`DropArena`] which allocated it. A
//! [`DropBox<T>`] can be consumed by its creator [`DropArena<T>`], which frees up the memory so that the
//! [`DropArena<T>`] can reuse it on another allocation and either returns or drops the underlying `T`.

use std::borrow::{Borrow, BorrowMut};
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::{mem, ptr};

use typed_arena::Arena;

/// An Item is either a free block, in which case it has a pointer to the next free block,
/// or it is occupied by a `T`.
union Item<T> {
    /// In the `pointer` variant, this [`Item<T>`] is a node in a linked list of [`Item<T>`]s,
    /// all of which are in the `pointer` variant. The pointer is semantically a [`&'arena mut T`], where
    /// where `&'arena` is the lifetime of the arena the [`Item`] occurs within; it should have the
    /// same no-alias guarantees. Note that if we have two [`DropArena<'arena, T>`]s, their free
    /// lists can "cross over" in the sense that arena 1's free list can include blocks originally
    /// allocated by 2 and vice versa, but their lists cannot overlap because of the no-alias
    /// invariant.
    pointer: Option<NonNull<Item<T>>>,

    /// In the `item`, the `T`'s lifetime is controlled by a `DropBox<T>`.
    item: ManuallyDrop<T>,
}

impl<T> Item<T> {
    #[inline]
    fn new(value: T) -> Self {
        Self {
            item: ManuallyDrop::new(value),
        }
    }
}

/// A ZST which invariantly depends on `'b`.
type Invariant<'b> = PhantomData<*mut &'b i32>;

/// An owning pointer to a `T` which has been allocated with a [`DropArena<'arena, T>`].
#[repr(transparent)]
// safety: a [`DropBox<'arena, T>`] must be transmutable to a [`&'arena mut T`]. *self.pointer
// must be an `Item` in the `item` configuration, with an undropped `T`.
pub struct DropBox<'arena, T> {
    pointer: &'arena mut Item<T>,
    _phantom: Invariant<'arena>,
}

impl<'arena, T> Deref for DropBox<'arena, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        // Safety: it's an invariant that self.pointer is always in the item configuration
        // with an undropped T.
        unsafe { &self.pointer.item }
    }
}

impl<'arena, T> Borrow<T> for DropBox<'arena, T> {
    #[inline]
    fn borrow(&self) -> &T {
        self.deref()
    }
}

impl<'arena, T> DerefMut for DropBox<'arena, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // Safety: it's an invariant that self.pointer is always in the item configuration
        // with an undropped T.
        unsafe { &mut self.pointer.item }
    }
}

impl<'arena, T> BorrowMut<T> for DropBox<'arena, T> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut T {
        self.deref_mut()
    }
}

impl<'arena, T> DropBox<'arena, T> {
    /// Safety: after this function is called, we can't use our [`DropBox`] as a reference to
    /// the underlying `T` again. This includes not calling [`drop`] on `self`.
    #[inline]
    unsafe fn drop_inner(&mut self) {
        ptr::drop_in_place::<T>(self.deref_mut())
    }

    /// This function returns the underlying `T` without freeing the memory used to allocate
    /// the `T`. The memory used to allocate the underlying `T` will be freed when the underlying
    /// `DropArena` is freed. To also free the memory, see `DropArena::box_to_inner`.
    #[inline]
    pub fn into_inner(mut self) -> T {
        // Safety: we don't use `self` again after the call to `take_inner`.
        unsafe {
            let result = self.take_inner();
            mem::forget(self);
            result
        }
    }

    /// Leaks `self`. The underlying `T` will never be dropped, nor will the memory allocated to
    /// store the `T` be reclaimed by the [`DropArena<T>`]. Only once the [`DropArena<T>`] is itself dropped
    /// will the memory be returned to the runtime.
    #[inline]
    pub fn leak(self) -> &'arena mut T {
        // Safety: we must use transmute to avoid violating stacked borrows. We know `self` is a
        // #[repr(transparent)] wrapper around `&'arena mut Item<T>`. Furthermore, we know that
        // self.pointer points to an Item in the item configuration with an undropped T.
        unsafe {
            let item_ptr: &'arena mut Item<T> = mem::transmute(self);
            &mut item_ptr.item
        }
    }

    /// Safety: after this function is called, we can't use our `DropBox` as a reference to
    /// the underlying `T` again. This includes not calling `Drop`.
    #[inline]
    unsafe fn take_inner(&mut self) -> T {
        ManuallyDrop::take(&mut self.pointer.item)
    }
}

/// Note that dropping the `DropBox<'a, 'b, T>` will not free the memory used to allocate
/// the `T`, but it will drop the `T`.
impl<'a, T> Drop for DropBox<'a, T> {
    #[inline]
    fn drop(&mut self) {
        // Safety: we can't use `self` at all after this.
        unsafe { self.drop_inner() }
    }
}

/// Our `DropArena` depends invariantly on the lifetime `'arena`. This allows us to ensure that
/// the memory allocated to a `DropBox` can only be reclaimed by the `DropArena` that created it,
/// since only that `DropArena` can be proved by the compiler to have the same `'arena` lifetime
/// parameter.
///
/// Note that `DropArena` is currently inefficient but functional for ZSTs. I don't know why you'd
/// want to use it for ZSTs, but if you do, be aware `DropArena` will allocate in this case.
pub struct DropArena<'arena, T> {
    arena: Arena<Item<T>>,
    start: Cell<Option<NonNull<Item<T>>>>,
    _phantom: Invariant<'arena>,
}

// TODO: make `DropArena` work efficiently on ZSTs.

impl<'arena, T> DropArena<'arena, T> {
    /// This function drops the underlying `T` in place and frees the memory. Note that
    /// calling `std::mem::drop(x)` will drop the underlying `T` but will *not* free the memory
    /// used to allocate the `T`; the memory will eventually be freed when we drop the `DropArena`.
    #[inline]
    pub fn drop_box(&'arena self, mut x: DropBox<'arena, T>) {
        // Safety: we don't use `x` to refer to `T` after calling `drop_inner`.
        unsafe {
            x.drop_inner();
        }
        self.free_without_dropping(x);
    }

    /// Immediately frees up the memory the arena used to allocate the `DropBox<T>`, but without
    /// calling `Drop::drop()` on the `T`.
    #[inline]
    pub fn free_without_dropping(&'arena self, ptr: DropBox<'arena, T>) {
        unsafe {
            // Safety: a DropBox<'arena, T> is representationally equivalent to a &'arena mut Item<T>,
            // which is representationally equivalent to a NonNull<Item<T>>
            let mut ptr: NonNull<Item<T>> = mem::transmute(ptr);
            // Safety: ptr came from a mutable reference with a 'arena lifetime, so it's valid
            // for that whole lifetime.
            ptr.as_mut().pointer = self.start.replace(Some(ptr));
        }
    }

    /// This function produces the underlying `T` from a [`DropBox<T>`], freeing the
    /// associated memory.
    #[inline]
    pub fn box_to_inner(&'arena self, mut x: DropBox<'arena, T>) -> T {
        unsafe {
            let result = x.take_inner();
            self.free_without_dropping(x);
            result
        }
    }

    #[inline]
    fn from_arena(arena: Arena<Item<T>>) -> Self {
        DropArena {
            arena,
            start: Cell::new(None),
            _phantom: Default::default(),
        }
    }

    /// Allocates `value` in our arena.
    #[inline]
    pub fn alloc(&'arena self, value: T) -> DropBox<'arena, T> {
        let item = Item::new(value);
        DropBox {
            pointer: match self.start.get() {
                None => self.arena.alloc(item),
                Some(mut pointer) => unsafe {
                    self.start.set(pointer.as_ref().pointer);
                    let pointer = pointer.as_mut();
                    *pointer = item;
                    pointer
                },
            },
            _phantom: Default::default(),
        }
    }

    /// Computes the total number of items that have been allocated, minus the number that have been
    /// deallocated. This function is slow, so it should only be called infrequently.
    /// Note that if this arena is calling [`DropArena::drop_box`] on boxes allocated by another
    /// [`DropArena`] of the same lifetime, the answer could be negative.
    #[inline]
    pub fn len(&'arena self) -> isize {
        let mut next = self.start.get();
        let mut count = 0;
        while let Some(ptr) = next {
            count += 1;
            unsafe { next = ptr.as_ref().pointer }
        }
        assert!(count <= isize::MAX as usize);
        let len = self.arena.len();
        assert!(len <= isize::MAX as usize);
        (self.arena.len() as isize) - (count as isize)
    }

    /// Returns the highest value `self.len()` has ever been over the entire life of `self`
    /// up until the moment this call is made.
    #[inline]
    pub fn max_len(&self) -> usize {
        self.arena.len()
    }

    /// Produces a new `DropArena`.
    #[inline]
    pub fn new() -> Self {
        Self::from_arena(Arena::new())
    }

    /// Exactly like [`Self::new`], except that we allocate a fixed starting capacity. Prefer
    /// to use [`with_capacity`].
    #[inline]
    pub fn new_with_capacity(n: usize) -> Self {
        Self::from_arena(Arena::with_capacity(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{random, thread_rng, Rng};
    use std::num::Wrapping;

    struct Node<'arena, T> {
        item: T,
        rest: Ptr<'arena, T>,
    }

    type Ptr<'arena, T> = Option<DropBox<'arena, Node<'arena, T>>>;

    struct List<'arena, T> {
        arena: &'arena DropArena<'arena, Node<'arena, T>>,
        ptr: Ptr<'arena, T>,
    }

    impl<'arena, T> Drop for List<'arena, T> {
        fn drop(&mut self) {
            while let Some(mut nxt) = self.ptr.take() {
                self.ptr = nxt.rest.take();
            }
        }
    }

    impl<'d, T> List<'d, T> {
        fn push(&mut self, val: T) {
            self.ptr = Some(self.arena.alloc(Node {
                item: val,
                rest: self.ptr.take(),
            }))
        }

        fn pop(&mut self) -> Option<T> {
            let first = self.ptr.take()?;
            let node = self.arena.box_to_inner(first);
            self.ptr = node.rest;
            Some(node.item)
        }

        fn into_vec(mut self) -> Vec<T> {
            let mut vec = vec![];
            while let Some(t) = self.pop() {
                vec.push(t);
            }
            vec
        }

        fn new(arena: &'d DropArena<'d, Node<'d, T>>) -> Self {
            Self { arena, ptr: None }
        }
    }

    #[test]
    fn test_linked_list() {
        let v1 = {
            let arena = &DropArena::new();
            let mut list = List::new(arena);
            for i in 0..100 {
                list.push(i);
            }
            list.into_vec()
        };

        let v2: Vec<_> = (0..100).rev().collect();
        assert_eq!(v1, v2);

        {
            let arena = &DropArena::new();
            let mut list = List::new(arena);

            for i in 0..100 {
                list.push(i);
            }

            assert_eq!(arena.arena.len(), 100);
            assert_eq!(arena.len(), 100);

            for i in (50..100).rev() {
                assert_eq!(i, list.pop().unwrap());
                assert_eq!(arena.len(), i);
                assert_eq!(arena.arena.len(), 100);
            }

            for i in 50..100 {
                list.push(i);
                assert_eq!(arena.len(), i + 1);
                assert_eq!(arena.arena.len(), 100);
            }

            println!("Finished with Tester::use_arena");
        };
    }

    #[test]
    fn simple() {
        let arena = &DropArena::new();
        let b = arena.alloc(5);
        arena.drop_box(b);
        let _b2 = arena.alloc(6);
    }

    #[test]
    fn test_box_functions() {
        {
            let arena = &DropArena::new();
            let b = arena.alloc(5);
            let r = b.leak();
            *r += 1;
            let b2 = arena.alloc(10);
            *r += b2.into_inner();
        }
    }

    #[test]
    fn test_nested() {
        assert_eq!(
            {
                let arena = &DropArena::new();
                let mut b1 = arena.alloc(5);
                let r = {
                    let arena1 = arena;
                    let arena2 = &DropArena::new();
                    let b2 = arena2.alloc(6);
                    *b1 += 100;
                    assert_eq!(arena1.box_to_inner(b1), 105);
                    // arena1.drop_box(b2); // Doesn't compile, even without the next line.
                    let r = arena2.box_to_inner(b2);
                    assert_eq!(arena2.len(), 0);
                    r
                };

                assert_eq!(arena.len(), 0);
                r
            },
            6
        );
    }

    #[test]
    fn safe_overlap() {
        let arena1 = &DropArena::new();
        let arena2 = &DropArena::new();
        let b1 = arena1.alloc("hello");
        let b2 = arena2.alloc("goodbye");
        arena1.drop_box(b2);
        arena2.drop_box(b1);
    }

    fn pop_random<T>(vec: &mut Vec<T>) -> T {
        assert!(!vec.is_empty());
        let idx = thread_rng().gen_range(0..vec.len());
        let max = vec.len() - 1;
        vec.swap(idx, max);
        vec.pop().unwrap()
    }

    #[test]
    fn test_random() {
        let max_len = 300;
        let min_len = 100;
        let start = (min_len + max_len) / 2;

        let arena: DropArena<[Wrapping<usize>; 3]> = DropArena::new();
        let mut vec = Vec::with_capacity(max_len);
        let mut sum: Wrapping<usize> = Wrapping(0);
        for _ in 0..((min_len + max_len) / 2) {
            vec.push(arena.alloc(random()));
        }

        for _ in 0..(5 * start) {
            let l = vec.len();
            if l == max_len || (l != min_len && random()) {
                sum += arena
                    .box_to_inner(pop_random(&mut vec))
                    .into_iter()
                    .sum::<Wrapping<usize>>();
            } else {
                vec.push(arena.alloc(random()));
            };
        }

        assert!(sum.0 > 0); // This has only a (1 / (1 + usize::MAX)) chance of failing.
        assert!(arena.max_len() <= max_len);
    }
}
