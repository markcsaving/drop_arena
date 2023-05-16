use std::borrow::{Borrow, BorrowMut};
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::{ManuallyDrop};
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::{mem, ptr};

use typed_arena::Arena;

union Item<T> {
    pointer: Option<NonNull<Item<T>>>,
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

/// This type is roughly equivalent to a `&'b mut T`. It has an artificial invariant relationship
/// with the lifetime `'a`. Can only be created using a `DropManager`.
#[repr(transparent)]
pub struct DropBox<'arena, T> {
    pointer: &'arena mut Item<T>,
    _phantom: Invariant<'arena>,
}

impl<'arena, T> Deref for DropBox<'arena, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
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
    /// Safety: after this function is called, we can't use our `DropBox` as a reference to
    /// the underlying `T` again. This includes not calling `drop` on `self`.
    #[inline]
    unsafe fn drop_inner(&mut self) {
        ptr::drop_in_place::<T>(self.deref_mut())
    }

    /// This function returns the underlying `T` without freeing the memory used to allocate
    /// the `T`. The memory used to allocate the underlying `T` will be freed when the underlying
    /// `DropArena` is freed. To also free the memory, see `DropArena::box_to_inner`.
    #[inline]
    pub fn to_inner(mut self) -> T {
        // Safety: we don't use `self` again after the call to `take_inner`.
        unsafe { self.take_inner() }
    }

    #[inline]
    pub fn leak(mut self) -> &'arena mut T {
        // Safety: we don't use `self` again after the call to `take_inner`.
        unsafe {
            let result = (self.deref_mut() as *mut T).as_mut().unwrap();
            mem::forget(self);
            result
        }
    }

    /// Safety: after this function is called, we can't use our `DropBox` as a reference to
    /// the underlying `T` again.
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

/// An object that can use a `DropArena` safely. I would strongly prefer not needing this trait
/// and instead using closures, but this is not yet possible.
pub trait ArenaUser {
    type Item<'arena>
    where
        Self: 'arena;
    type Output;

    fn use_arena<'arena>(
        self,
        arena: &'arena DropArena<'arena, Self::Item<'arena>>,
    ) -> Self::Output
    where
        Self: 'arena;
}

// TODO: make `DropArena` work efficiently on ZSTs.

impl<'arena, T> DropArena<'arena, T> {
    /// This function drops the underlying `T` in place and frees the memory. Note that
    /// calling `std::mem::drop(x)` will drop the underlying `T` but will *not* free the memory
    /// used to allocate the `T`; the memory will eventually be freed when we drop the `DropArena`.
    #[inline]
    pub fn drop_box(&'arena self, mut x: DropBox<'arena, T>) {
        // Safety: we don't use `x` to refer to `T` after calling `drop_inner`.
        // Safety: we know `x` was allocated with `self` since the `'arena` lifetimes match.
        unsafe {
            x.drop_inner();
            self.add_free_item(x);
        }
    }

    /// May only be called if `ptr` is a pointer to an `Item` allocated by `self.arena`,
    /// and if `ptr` is not in the linked list of free items already, and if there are no
    /// references to `ptr.item` floating around.
    #[inline]
    unsafe fn add_free_item(&'arena self, ptr: DropBox<'arena, T>) {
        let mut ptr: NonNull<Item<T>> = mem::transmute(ptr);
        ptr.as_mut().pointer = self.start.replace(Some(ptr))
    }

    /// This function produces the underlying `T` from a `ManualDropBox<T>`, freeing the
    /// associated memory.
    #[inline]
    pub fn box_to_inner(&'arena self, mut x: DropBox<'arena, T>) -> T {
        unsafe {
            let result = x.take_inner();
            self.add_free_item(x);
            result
        }
    }

    #[inline]
    fn new(arena: Arena<Item<T>>) -> Self {
        DropArena {
            arena,
            start: Cell::new(None),
            _phantom: Default::default(),
        }
    }

    // /// Allocates a `DropArena<F::Item>` with capacity `n` and calls `cont` on it.
    // #[inline]
    // pub fn with_capacity<R, F: DummyFamily>(
    //     cont: impl for<'c> FnOnce(DropArena<'c, F::Item<'c>>) -> R,
    //     n: usize,
    // ) -> R {
    //     cont(DropArena::new(Arena::with_capacity(n)))
    // }

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

    /// Computes the total number of items that have been allocated and not deallocated. This function
    /// is slow, so it should only be called infrequently.
    #[inline]
    pub fn len(&self) -> usize {
        let mut next = self.start.get();
        let mut count = 0;
        while let Some(ptr) = next {
            count += 1;
            unsafe { next = ptr.as_ref().pointer }
        }
        self.arena.len() - count
    }
}

/// Allocates a `DropArena<F::Item>` and calls `f.use_arena` on it.
#[inline]
pub fn with_new<R, F: ArenaUser<Output = R>>(f: F) -> R {
    f.use_arena(&DropArena::new(Arena::new()))
}

/// Allocates a `DropArena<F::Item>` with capacity `n` and calls `f.use_arena` on it.
#[inline]
pub fn with_capacity<R, F: ArenaUser<Output = R>>(f: F, n: usize) -> R {
    f.use_arena(&DropArena::new(Arena::with_capacity(n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_linked_list() {
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
                while let Some(_) = self.pop() {}
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

        struct Maker;

        impl ArenaUser for Maker {
            type Item<'arena> = Node<'arena, i32>;
            type Output = Vec<i32>;

            fn use_arena<'arena>(
                self,
                arena: &'arena DropArena<'arena, Self::Item<'arena>>,
            ) -> Self::Output where Self: 'arena {
                let mut list = List::new(arena);
                for i in 0..100 {
                    list.push(i);
                }
                list.into_vec()
            }
        }

        let v1 = with_new(Maker);
        let v2: Vec<i32> = (0..100).rev().collect();
        assert_eq!(v1, v2);

        struct Tester;

        impl ArenaUser for Tester {
            type Item<'arena> = Node<'arena, usize>;
            type Output = ();

            fn use_arena<'arena>(self, arena: &'arena DropArena<'arena, Self::Item<'arena>>) -> Self::Output where Self: 'arena {
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
            }
        }

        with_capacity(Tester, 100)
    }

    #[test]
    fn simple() {
        struct Simple;

        impl ArenaUser for Simple {
            type Item<'arena>  = i32 where Self: 'arena;
            type Output = ();

            fn use_arena<'arena>(self, arena: &'arena DropArena<'arena, Self::Item<'arena>>) -> Self::Output where Self: 'arena {
                let b = arena.alloc(5);
                arena.drop_box(b);
                let _b2 = arena.alloc(6);
            }
        }

        with_new(Simple)
    }

    #[test]
    fn test_nested() {
        struct S1;
        impl ArenaUser for S1 {
            type Item<'arena> = i32;
            type Output = i32;

            fn use_arena<'arena>(
                self,
                arena: &'arena DropArena<'arena, Self::Item<'arena>>,
            ) -> Self::Output where Self: 'arena {

                struct S2<'arena>(&'arena DropArena<'arena, i32>);

                impl<'d1> ArenaUser for S2<'d1> {
                    type Item<'arena> = i32 where Self: 'arena;
                    type Output = i32;

                    fn use_arena<'d2>(
                        self,
                        arena2: &'d2 DropArena<'d2, Self::Item<'d2>>,
                    ) -> Self::Output
                    where
                        Self: 'd2,
                    {
                        let arena1 = self.0;
                        let mut b1 = arena1.alloc(5);
                        let b2 = arena2.alloc(6);
                        *b1 += 100;
                        assert_eq!(arena1.box_to_inner(b1), 105);
                        // arena1.drop_box(b2); // Doesn't compile
                        let r = arena2.box_to_inner(b2);
                        assert_eq!(arena2.len(), 0);
                        r
                    }
                }


                let r = with_new(S2(arena));
                assert_eq!(arena.len(), 0);
                r
            }
        }


        assert_eq!(with_new(S1), 6);
    }
}
