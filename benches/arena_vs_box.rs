use criterion::{criterion_group, criterion_main, Criterion};
use drop_arena::*;
use rand::*;
use typed_arena::*;

pub fn leak_all(c: &mut Criterion) {
    let mut group = c.benchmark_group("only-allocs");

    group.bench_function("box", |b| {
        b.iter(|| {
            for i in 0..500 {
                std::mem::forget(std::hint::black_box(Box::new(i)));
            }
        })
    });

    group.bench_function("dropbox", |b| {
        b.iter(|| {
            let arena = DropArena::new();
            for i in 0..500 {
                std::mem::forget(std::hint::black_box(arena.alloc(i)));
            }
        })
    });

    group.bench_function("typed-arena-ref", |b| {
        b.iter(|| {
            let arena = Arena::new();
            for i in 0..500 {
                std::mem::forget(std::hint::black_box(arena.alloc(i)));
            }
        })
    });

    group.finish();
}

pub fn alloc_drop(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc-drop");

    group.bench_function("box", |b| {
        b.iter(|| {
            for i in 0..500 {
                std::hint::black_box(Box::new(i));
            }
        })
    });

    group.bench_function("dropbox", |b| {
        b.iter(|| {
            let arena = DropArena::new();
            for i in 0..500 {
                std::hint::black_box((&arena, arena.alloc(i)));
            }
        })
    });

    group.bench_function("typed-arena-ref", |b| {
        b.iter(|| {
            let arena = Arena::new();
            for i in 0..500 {
                std::hint::black_box((&arena, arena.alloc(i)));
            }
        })
    });

    group.finish();
}

fn random_array() -> [usize; 3] {
    random()
}

fn pop_random<T>(vec: &mut Vec<T>) -> T {
    assert!(!vec.is_empty());
    let idx = thread_rng().gen_range(0..vec.len());
    let max = vec.len() - 1;
    vec.swap(idx, max);
    vec.pop().unwrap()
}

pub fn random_add_drop(c: &mut Criterion) {
    let max_len = 5000;
    let min_len = 1000;
    let start = (min_len + max_len) / 2;

    let mut group = c.benchmark_group("random-add-drop");

    group.bench_function("box", |b| {
        b.iter(|| {
            let mut vec = Vec::with_capacity(max_len);
            let mut sum: usize = 0;
            for _ in 0..start {
                vec.push(Box::new(random_array()));
            }

            for _ in 0..(10 * start) {
                let l = vec.len();
                if l == max_len || (l != min_len && random()) {
                    sum += pop_random(&mut vec).into_iter().sum::<usize>();
                } else {
                    vec.push(Box::new(random_array()));
                };
            }
            sum
        })
    });

    group.bench_function("dropbox", |b| {
        b.iter(|| {
            let arena = DropArena::new();
            let mut vec = Vec::with_capacity(max_len);
            let mut sum: usize = 0;
            for _ in 0..start {
                vec.push(arena.alloc(random_array()));
            }

            for _ in 0..(10 * start) {
                let l = vec.len();
                if l == max_len || (l != min_len && random()) {
                    sum += arena
                        .box_to_inner(pop_random(&mut vec))
                        .into_iter()
                        .sum::<usize>();
                } else {
                    vec.push(arena.alloc(random_array()));
                };
            }
            sum
        })
    });

    group.finish();
}

criterion_group!(benches, leak_all, alloc_drop, random_add_drop);
criterion_main!(benches);
