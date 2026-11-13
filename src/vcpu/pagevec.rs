//! A per-page table for a large, sparse address space.
//!
//! A guest reserves far more address space than it touches, and every `fork`
//! needs its own copy of the per-page tables. A flat `Vec` costs one entry per
//! page of *reservable* address space on every fork, whether used or not (tens
//! of MiB per process, outside any guest limit). This stores runs of pages in
//! chunks that exist only once written and are shared copy-on-write, so a clone
//! costs a small top-level table and memory follows what the guest actually uses.

use std::ops::{Index, IndexMut};
use std::sync::Arc;

/// Pages per chunk: 16 MiB of address space.
const CHUNK: usize = 4096;

/// A table of `len` values of `T`, all initially `default`.
#[derive(Debug, Clone)]
pub(crate) struct PageVec<T> {
    chunks: Vec<Option<Arc<[T; CHUNK]>>>,
    len: usize,
    default: T,
}

impl<T: Copy + PartialEq> PageVec<T> {
    pub(crate) fn new(len: usize, default: T) -> Self {
        Self {
            chunks: vec![None; len.div_ceil(CHUNK)],
            len,
            default,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The entry for page `p`, if `p` is in range.
    pub(crate) fn get(&self, p: usize) -> Option<&T> {
        (p < self.len).then(|| &self[p])
    }

    /// Reset every entry to the default, freeing all memory behind them.
    pub(crate) fn clear(&mut self) {
        self.chunks = Vec::new();
    }

    /// Pages per chunk: what [`Self::uniform_chunk`] answers for.
    pub(crate) const CHUNK: usize = CHUNK;

    /// The value every page of chunk `c` (pages `c * CHUNK ..`) holds, if
    /// they all hold one: the default for a chunk never written.
    pub(crate) fn uniform_chunk(&self, c: usize) -> Option<T> {
        match self.chunks.get(c) {
            Some(Some(chunk)) => {
                let first = chunk[0];
                chunk.iter().all(|v| *v == first).then_some(first)
            }
            _ => Some(self.default),
        }
    }

    /// The pages holding something other than the default, in order. Only
    /// visits chunks that exist, so it doesn't scan the whole address space.
    pub(crate) fn non_default(&self) -> impl Iterator<Item = (usize, T)> + '_ {
        self.chunks.iter().enumerate().flat_map(move |(c, chunk)| {
            chunk.iter().flat_map(move |chunk| {
                chunk
                    .iter()
                    .enumerate()
                    .filter(move |(_, v)| **v != self.default)
                    .map(move |(i, v)| (c * CHUNK + i, *v))
            })
        })
    }
}

impl<T> Index<usize> for PageVec<T> {
    type Output = T;

    fn index(&self, p: usize) -> &T {
        assert!(p < self.len, "page {p} out of range");
        match self.chunks.get(p / CHUNK) {
            Some(Some(chunk)) => &chunk[p % CHUNK],
            _ => &self.default,
        }
    }
}

impl<T: Copy> IndexMut<usize> for PageVec<T> {
    fn index_mut(&mut self, p: usize) -> &mut T {
        assert!(p < self.len, "page {p} out of range");
        // After `clear`, writes rebuild the table on demand.
        if self.chunks.is_empty() {
            self.chunks.resize(self.len.div_ceil(CHUNK), None);
        }
        let default = self.default;
        let chunk = self.chunks[p / CHUNK].get_or_insert_with(|| Arc::new([default; CHUNK]));
        &mut Arc::make_mut(chunk)[p % CHUNK]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwritten_pages_read_as_the_default_and_cost_nothing() {
        let v: PageVec<u8> = PageVec::new(1 << 24, 7);
        assert_eq!(v[12_345], 7);
        assert_eq!(v.get(1 << 24), None);
        assert!(v.chunks.iter().all(Option::is_none));
    }

    #[test]
    fn a_clone_is_independent_and_shares_until_written() {
        let mut a: PageVec<u8> = PageVec::new(CHUNK * 4, 0);
        a[5] = 1;
        let mut b = a.clone();
        let chunk = |v: &PageVec<u8>| v.chunks[0].as_ref().unwrap().clone();
        assert!(Arc::ptr_eq(&chunk(&a), &chunk(&b)), "shared after clone");
        b[6] = 2;
        assert!(!Arc::ptr_eq(&chunk(&a), &chunk(&b)), "split on write");
        assert_eq!((a[5], a[6]), (1, 0));
        assert_eq!((b[5], b[6]), (1, 2));
    }

    #[test]
    fn clear_frees_everything_and_the_table_still_works() {
        let mut v: PageVec<u8> = PageVec::new(CHUNK * 4, 0);
        v[CHUNK * 3] = 9;
        v.clear();
        assert_eq!(v.chunks.len(), 0);
        assert_eq!(v[CHUNK * 3], 0);
        assert_eq!(v.len(), CHUNK * 4);
        v[1] = 3;
        assert_eq!(v[1], 3);
    }

    #[test]
    fn non_default_lists_only_what_was_set() {
        let mut v: PageVec<u8> = PageVec::new(CHUNK * 4, 0);
        v[3] = 1;
        v[CHUNK * 2 + 1] = 2;
        v[3 + 1] = 5;
        v[3 + 1] = 0;
        let got: Vec<_> = v.non_default().collect();
        assert_eq!(got, vec![(3, 1), (CHUNK * 2 + 1, 2)]);
    }
}
