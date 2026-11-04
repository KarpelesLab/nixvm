//! The pages an address space has locked with `mlock`/`mlockall`, kept as
//! disjoint page ranges so locking a large region costs one entry.
//!
//! Nothing here is ever paged out, so a lock changes nothing about the memory.
//! It only exists so `RLIMIT_MEMLOCK` can be charged and enforced the way Linux
//! does: by the number of locked pages.

use std::collections::BTreeMap;

/// Disjoint, non-adjacent `[start, end)` page ranges, keyed by start.
#[derive(Clone, Debug, Default)]
pub(crate) struct PageLocks {
    ranges: BTreeMap<usize, usize>,
    pages: usize,
    /// `mlockall(MCL_FUTURE)`: mappings made from now on are locked too.
    pub(crate) future: bool,
}

impl PageLocks {
    /// How many pages are locked.
    pub(crate) fn pages(&self) -> usize {
        self.pages
    }

    /// How many pages of `[start, end)` are locked.
    pub(crate) fn locked_within(&self, start: usize, end: usize) -> usize {
        let mut n = 0;
        // The range starting at or before `start` may reach into it.
        if let Some((_, &e)) = self.ranges.range(..=start).next_back() {
            n += e.min(end).saturating_sub(start);
        }
        for (&s, &e) in self.ranges.range(start + 1..end) {
            n += e.min(end) - s;
        }
        n
    }

    /// Lock `[start, end)`.
    pub(crate) fn insert(&mut self, mut start: usize, mut end: usize) {
        if start >= end {
            return;
        }
        // Absorb every range overlapping or touching `[start, end)`.
        if let Some((&s, &e)) = self.ranges.range(..=start).next_back()
            && e >= start
        {
            start = s;
            end = end.max(e);
            self.pages -= e - s;
            self.ranges.remove(&s);
        }
        let touching: Vec<(usize, usize)> = self
            .ranges
            .range(start..=end)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in touching {
            end = end.max(e);
            self.pages -= e - s;
            self.ranges.remove(&s);
        }
        self.pages += end - start;
        self.ranges.insert(start, end);
    }

    /// Unlock `[start, end)`.
    pub(crate) fn remove(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let mut hit: Vec<(usize, usize)> = Vec::new();
        if let Some((&s, &e)) = self.ranges.range(..start).next_back()
            && e > start
        {
            hit.push((s, e));
        }
        hit.extend(self.ranges.range(start..end).map(|(&s, &e)| (s, e)));
        for (s, e) in hit {
            self.ranges.remove(&s);
            self.pages -= e - s;
            if s < start {
                self.ranges.insert(s, start);
                self.pages += start - s;
            }
            if e > end {
                self.ranges.insert(end, e);
                self.pages += e - end;
            }
        }
    }

    /// Unlock everything, `munlockall` and `execve`.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::PageLocks;

    #[test]
    fn ranges_merge_split_and_count() {
        let mut l = PageLocks::default();
        l.insert(10, 20);
        l.insert(20, 25); // touching: merges
        l.insert(5, 12); // overlapping
        assert_eq!(l.pages(), 20);
        assert_eq!(l.ranges.len(), 1);
        assert_eq!(l.locked_within(0, 8), 3);
        assert_eq!(l.locked_within(22, 100), 3);
        l.remove(10, 15); // splits
        assert_eq!(l.pages(), 15);
        assert_eq!(l.ranges.len(), 2);
        assert_eq!(l.locked_within(0, 100), 15);
        l.insert(40, 50);
        l.remove(0, 45);
        assert_eq!(l.pages(), 5);
        assert_eq!(l.locked_within(45, 50), 5);
        l.insert(0, 100);
        assert_eq!(l.pages(), 100);
        assert_eq!(l.ranges.len(), 1);
        l.clear();
        assert_eq!(l.pages(), 0);
    }
}
