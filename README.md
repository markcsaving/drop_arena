# Introduction 
A `DropArena<T>` can allocate or deallocate individual elements of type `T`. Only allocating elements of a fixed size 
and alignment allows the allocator to be extremely efficient compared to an ordinary implementation of `malloc` and `free`.
Think of `DropArena` as providing a combination of the functionality of an `Arena` and the allocator that makes `Box`es.

The `DropArena` can return a `DropBox<T>`, which functions very much like a `Box<T>` except for being tied to the 
`DropArena` that allocated it. A `DropBox` can be used exactly like a `&mut T`; in fact, it is a `repr(transparent)`
wrapper around a `&mut T`. 

When it comes to getting rid of a `DropBox<T>`, there are several options. First, you may use `std::mem::drop` (or let the 
`DropBox` go out of scope). This will call `drop` on the underlying `T`, but it will *not* reclaim the memory needed
to allocate the `T`. Similarly, you may use `DropBox::to_inner`, which extracts the underlying `T` but does not reclaim
the memory the `T` formerly occupied. This memory will eventually be reclaimed when the `DropArena<T>` is itself dropped.
Finally, you may call `DropBox::leak`, which produces a `&mut T`. This means that unless unsafe code is used, the `T` will
never be `drop`ed. However, when the `DropArena` which allocated the `DropBox` is dropped, the memory will be reclaimed.

In order to reclaim the memory allocated to a `DropBox<T>`, we need a reference to the `DropArena<T>` which allocated it.
We can use the `DropArena::drop_box` method on a `DropBox` to drop the underlying value and reclaim the memory. We can 
also use the `DropArena::box_to_inner` method to retrieve the underlying `T` from a `DropBox<T>` and reclaim the memory
it used.

To guarantee that an arena can only reclaim memory from `DropBox`es it allocated (or one allocated by a drop arena with 
exactly the same lifetime), we need to use lifetime magic. A `DropArena` is tagged with the lifetime it will live, and
it has an invariant relationship with this lifetime. `DropBox`es have an invariant relationship with the lifetime of 
the `DropArena` that created them. 

It is not recommended to have multiple `DropArena<T>`s with the same lifetime. In particular, if arena 1 keeps allocating
`DropBox<T>`s which arena 2 keeps consuming, you won't get any benefit out of reclaiming the memory. However, it is 
perfectly safe to do this.