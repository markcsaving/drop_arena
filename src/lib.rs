#![warn(missing_docs)]
#![cfg_attr(not(any(feature = "std", test)), no_std)]

// After modifying the below, please make sure to run
// $ cargo readme > README-utf16.md
//! # Introduction
//! A [`DropArena<T>`] can allocate or deallocate individual elements of type `T`. Only allocating elements of a fixed size
//!! and alignment allows the allocator to be extremely efficient compared to an ordinary implementation of `malloc` and `free`.
//! Think of [`DropArena`] as providing a combination of the functionality of an [`Arena`] and the allocator that makes Boxes.
//! 
//! The [`DropArena`] can return a [`DropBox<T>`], which functions very much like a [`Box<T>`] except for being tied to the
//! [`DropArena`] that allocated it. A [`DropBox`] can be used exactly like a [`&mut T`]; in fact, it is a `repr(transparent)`
//! wrapper around a [`&mut T`]. 
//! 
//! When it comes to getting rid of a [`DropBox<T>`], there are several options. First, you may use [`drop`] (or let the 
//! [`DropBox`] go out of scope). This will call [`drop`] on the underlying `T`, but it will *not* reclaim the memory needed
//! to allocate the `T`. Similarly, you may use [`DropBox::into_inner`], which extracts the underlying `T` but does not reclaim
//! the memory the `T` formerly occupied. This memory will eventually be reclaimed when the [`DropArena<T>`] is itself dropped.
//! Finally, you may call [`DropBox::leak`], which produces a [`&mut T`]. This means that unless unsafe code is used, the `T` will
//! never be [`drop`]ed. However, when the [`DropArena`] which allocated the [`DropBox`] is dropped, the memory will be reclaimed.
//! 
//! In order to reclaim the memory allocated to a [`DropBox<T>`], we need a reference to the [`DropArena<T>`] which allocated it.
//! We can use the [`DropArena::drop_box`] method on a [`DropBox`] to drop the underlying value and reclaim the memory. We can 
//! also use the [`DropArena::box_into_inner`] method to retrieve the underlying `T` from a [`DropBox<T>`] and reclaim the memory
//! it used.
//! 
//! To guarantee that an arena can only reclaim memory from [`DropBox`]es it allocated (or one allocated by a drop arena with 
//! exactly the same lifetime), we need to use lifetime magic. A [`DropArena`] is tagged with the lifetime it will live, and
//! it has an invariant relationship with this lifetime. [`DropBox`]es have an invariant relationship with the lifetime of 
//! the [`DropArena`] that created them. 
//! 
//! It is not recommended to have multiple [`DropArena<T>`]s with the same lifetime. In particular, if arena 1 keeps allocating
//! [`DropBox<T>`]s which arena 2 keeps consuming, you won't get any benefit out of reclaiming the memory. However, it is 
//! perfectly safe to do this.
//! 
//! # Complexity
//! 
//! Calling [`DropArena::box_into_inner()`] or [`DropBox::into_inner()`] is O(1) with very small constants (except if 
//! the size of `T` is large - then copying the `T` dominates). The corresponding [`drop`] functions are also O(1) + the 
//! time the call to [`Drop::drop`] takes with small constants.
//! 
//! Allocating is also very fast. There are three possible paths for an allocation. First, the arena has a free space where
//! something was previously allocated. In this case, allocation is O(1) with small constants. Second, the preallocated
//! capacity of the Arena is large enough to fit one more element. In this case, allocation is O(1) with small constants.
//! Third, the arena has genuinely run out of space (this is the most uncommon case, even when we are only doing allocations
//! and no drops). In this case, we must allocate more space using the system allocator. We follow the same guidelines as 
//! [`typed_arena`], making a single allocation with enough space for many more `T`s (in fact, we actually implement 
//! [`DropArena`] using [`typed_arena`]).
//!
//! # Recursive owning data structures
//!
//! We can write some basic owning data structures using our arena as follows. The list implementation
//! below is inspired by [Learning Rust With Entirely Too Many Linked Lists](https://rust-unofficial.github.io/too-many-lists/index.html).
//!
//! ```
//! use drop_arena::{DropArena, DropBox};
//! struct Node<'arena, T> {
//!     item: T,
//!     rest: Link<'arena, T>,
//! }
//!
//! type Link<'arena, T> = Option<DropBox<'arena, Node<'arena, T>>>;
//!
//! struct List<'arena, T> {
//!     arena: &'arena DropArena<'arena, Node<'arena, T>>,
//!     ptr: Link<'arena, T>,
//! }
//!
//! impl<'arena, T> Drop for List<'arena, T> {
//!     fn drop(&mut self) {
//!         while let Some(mut nxt) = self.ptr.take() {
//!             self.ptr = nxt.rest.take();
//!             self.arena.drop_box(nxt);
//!         }
//!     }
//! }
//!
//! impl<'d, T> List<'d, T> {
//!     fn push(&mut self, val: T) {
//!         self.ptr = Some(self.arena.alloc(Node {
//!             item: val,
//!             rest: self.ptr.take(),
//!         }))
//!     }
//!
//!     fn pop(&mut self) -> Option<T> {
//!         let first = self.ptr.take()?;
//!         let node = self.arena.box_into_inner(first);
//!         self.ptr = node.rest;
//!         Some(node.item)
//!     }
//!
//!     fn new(arena: &'d DropArena<'d, Node<'d, T>>) -> Self {
//!         Self { arena, ptr: None }
//!     }
//! }
//!
//!
//! let arena = DropArena::new();
//! let mut list = List::new(&arena);
//!
//! for i in 0..100 {
//!     list.push(i);
//! }
//!
//! for i in (0..100).rev() {
//!     assert_eq!(list.pop(), i);
//! }
//! ```
//! 
//! # Areas of Improvement
//!
//! This allocator works for zero-sized types, but it is not efficient in this case. I plan to address this in the future 
//! using conditional types. The issue is that keeping a free block list requires pointers. However, in theory, when we are
//! dealing with ZSTs, we could just choose not to have a free list at all. I would like to separately implement a special
//! arena for ZSTs using [CondType](https://!github.com/nvzqz/condtype), but this crate is still limited. In order for it to be usable here, we need
//! [this issue](https://!github.com/rust-lang/project-const-generics/issues/26) to be resolved.
//! 
//! Much more testing is required to ensure that [`DropArena`]s are safe. I've done some elementary experimentation with Miri,
//! but exhaustive fuzzing is needed. This code uses a fair amount of `unsafe`, and that means there are plenty of chances for 
//! serious bugs to appear. I think I've caught most of them, but it's not impossible I neglected one.
//!
//! Making sure that a [`DropBox<T>`] implements all the traits that a [`Box<T>`] does seems desirable.
//!
//! # Maintenance
//!
//! This is my first open-source project, so I may not be able to find time to properly maintain it.
//! That said, I will do my best, time-permitting, especially for serious bugs. Please be patient.


use core::borrow::{Borrow, BorrowMut};
use core::cell::Cell;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::{mem, ptr};

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

/// An owning pointer to a `T` which has been allocated with a `DropArena<'arena, T>`.
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
    /// the underlying `T` again. This includes not calling [`DropBox::drop`] on `self`. Warning: this
    /// function can panic, but this function panicking doesn't relieve the caller of its
    /// obligations not to use `self` again as a reference to `T`, including via [`drop`].
    #[inline]
    unsafe fn drop_inner(&mut self) {
        ptr::drop_in_place::<T>(self.deref_mut())
    }

    /// This function returns the underlying `T` without freeing the memory used to allocate
    /// the `T`. The memory used to allocate the underlying `T` will be freed when the underlying
    /// [`DropArena`] is freed. To also free the memory, see [`DropArena::box_into_inner`].
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    /// {
    ///     let arena = DropArena::new();
    ///     let mut b = arena.alloc(100);
    ///     # assert_eq!(arena.len(), 1);
    ///     *b += 1;
    ///     assert_eq!(b.into_inner(), 101);
    ///     // The arena still thinks the allocated memory is in use here,
    ///     # assert_eq!(arena.len(), 1);
    /// } // but the memory is released to the system when arena goes out of scope
    /// ```
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
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// struct IntWrapper(i32);
    /// impl Drop for IntWrapper {
    ///     fn drop(&mut self) {
    ///         panic!("We should never drop an IntWrapper.")
    ///     }
    /// }
    ///
    /// {
    ///     let arena = DropArena::new();
    ///     let b = arena.alloc(IntWrapper(0));
    ///     assert_eq!(arena.len(), 1);
    ///
    ///     let b = b.leak();
    ///     # assert_eq!(arena.len(), 1);
    ///
    ///     b.0 += 1;
    ///     assert_eq!(b.0, 1);
    ///  } // The memory is reclaimed here, when the arena goes out of scope.
    /// // Note that we never drop the underlying IntWrapper.
    /// ```
    ///
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
    /// the underlying `T` again. This includes not calling `Drop`. This function cannot panic.
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

/// Our [`DropArena`] depends invariantly on its own lifetime `'arena`. This allows us to ensure
/// that [`DropBox`]s are collected by their corresponding [`DropArena`]s (or those of exactly
/// the same lifetime, which is safe but inefficient).
///
/// Note that `DropArena` is currently inefficient but functional for ZSTs. I don't know why you'd
/// want to use it for ZSTs, but if you do, be aware `DropArena` will allocate in this case.
pub struct DropArena<'arena, T> {
    arena: Arena<Item<T>>,
    start: Cell<Option<NonNull<Item<T>>>>,
    _phantom: Invariant<'arena>,
}

impl<'arena, T> DropArena<'arena, T> {
    /// This function drops the underlying `T` in place and frees the memory. Note that
    /// calling [`core::mem::drop(x)`] will drop the underlying `T` but will *not* free the memory
    /// used to allocate the `T`; the memory will eventually be freed when we drop the [`DropArena`].
    ///
    /// Note that calling [`Self::drop_box`] is more efficient than dropping the result of [`Self::box_into_inner`].
    /// It may be difficult or impossible for the compiler to optimize the latter into the former.
    ///
    /// # Example
    /// ```
    /// # use std::ops::Deref;
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::new();
    /// let b = arena.alloc(5);
    /// # let addr = b.deref() as *const _ as usize;
    /// # assert_eq!(arena.len(), 1);
    /// arena.drop_box(b);
    /// # assert_eq!(arena.len(), 0);
    /// // We can now allocate another integer in the same place as the last one.
    /// let b = arena.alloc(6);
    /// # assert_eq!(arena.max_len(), 1);
    /// # assert_eq!(addr, b.deref() as *const _ as usize);
    /// ```
    #[inline]
    pub fn drop_box(&'arena self, x: DropBox<'arena, T>) {
        // SAFETY: Wrapping x in a ManuallyDrop prevents us from running into issues if `T::drop` panics.
        // If `T::drop` does panic, the spot the `T` occupied will be leaked.
        let mut x = ManuallyDrop::new(x);
        unsafe { x.deref_mut().drop_inner() };
        self.free_without_dropping(ManuallyDrop::into_inner(x))
    }

    /// Immediately frees up the memory the arena used to allocate the [`DropBox<T>`], but without
    /// calling [`Drop::drop()`] on the `T`. It is the "opposite" of [`DropBox::leak`].
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// struct IntWrapper(i32);
    /// impl Drop for IntWrapper {
    ///     fn drop(&mut self) {
    ///         panic!("We should never drop an IntWrapper.")
    ///     }
    /// }
    ///
    /// let arena = DropArena::new();
    /// let b = arena.alloc(IntWrapper(3));
    /// # assert_eq!(arena.len(), 1);
    /// arena.free_without_dropping(b);
    /// # assert_eq!(arena.len(), 0);
    /// ```
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
    /// associated memory. Note that it is potentially less efficient to call this function and immediately
    /// drop the `T` than to simply call `drop_box`.
    ///
    /// # Example
    /// ```
    /// use drop_arena::{DropArena, DropBox};
    ///
    /// let arena = DropArena::new();
    /// let string: String = "hello".to_string();
    /// let string: DropBox<String> = arena.alloc(string);
    /// let string: String = arena.box_into_inner(string);
    /// // The arena is now free to reuse the slot that `string` took up.
    /// # assert_eq!(arena.len(), 0);
    /// # assert_eq!(string, "hello");
    /// ```
    #[inline]
    pub fn box_into_inner(&'arena self, mut x: DropBox<'arena, T>) -> T {
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
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::new();
    /// let b = arena.alloc(5);
    /// # assert_eq!(*b, 5)
    /// ```
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
    /// [`DropArena`] of the same lifetime, the answer could be negative. It is not recommended to have
    /// two arenas of the same type and lifetime; there is never a need for this as far as I can tell.
    ///
    /// # Example
    ///
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::new();
    /// for i in 0..10 {
    ///     assert_eq!(arena.len(), i);
    ///     arena.alloc(i);
    /// }
    /// assert_eq!(arena.len(), 10);
    /// let b = arena.alloc(10);
    /// assert_eq!(arena.len(), 11);
    /// arena.drop_box(b);
    /// assert_eq!(arena.len(), 10);
    /// ```
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
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::new();
    /// let mut boxes = Vec::new();
    /// for i in 0..10 {
    ///     boxes.push(arena.alloc(i));
    /// }
    /// for bx in boxes {
    ///     arena.drop_box(bx);
    /// }
    ///
    /// assert_eq!(arena.len(), 0);
    /// assert_eq!(arena.max_len(), 10);
    /// ```
    #[inline]
    pub fn max_len(&self) -> usize {
        self.arena.len()
    }

    /// Produces a new [`DropArena`] with the default size (around 1024 bytes of capacity).
    ///
    /// # Technical note
    /// Keep in mind that the [`DropArena`] is not laid out as a `[T]` if the size
    /// of T is less than the size of a `usize`, or if the alignment of a `T` is less than that
    /// of a `usize`. Thus, it is possible the default size is a little smaller than expected.
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::new();
    /// # arena.alloc(1);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::from_arena(Arena::new())
    }

    /// Exactly like [`Self::new`], except that we allocate a fixed starting capacity. Here,
    /// `n` is the number of elements we store. Be aware that calling this with `n = 0` is exactly
    /// the same as calling it with `n = 1`; we heap-allocate in either case.
    ///
    /// # Example
    /// ```
    /// use drop_arena::DropArena;
    ///
    /// let arena = DropArena::with_capacity(5000);
    ///
    /// for i in 0..5000 {
    ///     arena.alloc(i);
    /// }
    /// ```
    #[inline]
    pub fn with_capacity(n: usize) -> Self {
        Self::from_arena(Arena::with_capacity(n))
    }
}

#[cfg(test)]
mod tests {
    #[cfg(no_std)]
    extern crate alloc;
    use super::*;
    #[cfg(no_std)]
    use alloc::vec;
    #[cfg(no_std)]
    use alloc::vec::Vec;
    use core::num::Wrapping;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use rand::{random, thread_rng, Rng};
    use std::panic::catch_unwind;

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
                self.arena.drop_box(nxt);
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
            let node = self.arena.box_into_inner(first);
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

            assert_eq!(arena.max_len(), 100);
            assert_eq!(arena.len(), 100);

            for i in (50..100).rev() {
                assert_eq!(i, list.pop().unwrap());
                assert_eq!(arena.len(), i);
                assert_eq!(arena.max_len(), 100);
            }

            for i in 50..100 {
                list.push(i);
                assert_eq!(arena.len(), i + 1);
                assert_eq!(arena.max_len(), 100);
            }

            drop(list);
            assert_eq!(arena.len(), 0);

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
                    assert_eq!(arena1.box_into_inner(b1), 105);
                    // arena1.drop_box(b2); // Doesn't compile, even without the next line.
                    let r = arena2.box_into_inner(b2);
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

        for _ in 0..(3 * start) {
            let l = vec.len();
            if l == max_len || (l != min_len && random()) {
                sum += arena
                    .box_into_inner(pop_random(&mut vec))
                    .into_iter()
                    .sum::<Wrapping<usize>>();
            } else {
                vec.push(arena.alloc(random()));
            };
        }

        assert!(sum.0 > 0); // This has only a (1 / (1 + usize::MAX)) chance of failing.
        assert!(arena.max_len() <= max_len);
    }

    #[test]
    /// This test should be run with Miri. The purpose is to make sure that panicking [`Drop`]
    /// implementations don't cause undefined behaviour, potentially by running `drop` a second
    /// time. Because running drop a second time is undefined behaviour, this test may "pass"
    /// even if it should fail unless run with Miri.
    fn catch_panic() {
        static CELL: AtomicUsize = AtomicUsize::new(0);
        struct Dropping;

        impl Drop for Dropping {
            fn drop(&mut self) {
                CELL.fetch_add(1, Ordering::Relaxed);
                panic!("panic in Dropping::drop")
            }
        }

        let arena = &DropArena::new();
        let b = arena.alloc(Dropping);

        catch_unwind(core::panic::AssertUnwindSafe(|| arena.drop_box(b))).unwrap_err();
        assert_eq!(CELL.load(Ordering::Relaxed), 1);
        let b = arena.alloc(Dropping);
        catch_unwind(core::panic::AssertUnwindSafe(|| drop(b))).unwrap_err();
        assert_eq!(CELL.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn len_error() {
        let arena1 = &DropArena::new();
        {
            let arena2 = &DropArena::new();
            let b = arena2.alloc(1);
            arena1.drop_box(b);
        }
        // arena1.len() // this line must not compile
    }
}
