use drop_arena::*;
use typed_arena::*;
use criterion::{criterion_main, Criterion, criterion_group};

pub fn leak_all(c: &mut Criterion){
    let mut group = c.benchmark_group("only-allocs");

    group.bench_function("box", |b| b.iter(|| {
        for i in 0..500 {
            std::mem::forget(std::hint::black_box(Box::new(i)));
        }
    }));

    group.bench_function("dropbox", |b| b.iter(|| {
        let arena = DropArena::new();
        for i in 0..500 {
            std::mem::forget(std::hint::black_box(arena.alloc(i)));
        }
    }));

    group.bench_function("typed-arena-ref", |b| b.iter(|| {
        let arena = Arena::new();
        for i in 0..500 {
            std::mem::forget(std::hint::black_box(arena.alloc(i)));
        }
    }));

    group.finish();
}

pub fn alloc_drop(c: &mut Criterion) {
    let mut group = c.benchmark_group("only-allocs");

    group.bench_function("box", |b| b.iter(|| {
        for i in 0..500 {
            std::hint::black_box(Box::new(i));
        }
    }));

    group.bench_function("dropbox", |b| b.iter(|| {
        let arena = DropArena::new();
        for i in 0..500 {
            std::hint::black_box((&arena, arena.alloc(i)));
        }
    }));

    group.bench_function("typed-arena-ref", |b| b.iter(|| {
        let arena = Arena::new();
        for i in 0..500 {
            std::hint::black_box((&arena, arena.alloc(i)));
        }
    }));

    group.finish();
}

criterion_group!(benches, leak_all, alloc_drop);
criterion_main!(benches);