"""
EEC3414 알고리즘 H/W #1
세 가지 정렬 알고리즘 구현 모듈

구현 목록
1) Quick Sort 변종: Median-of-Three + 3-way partition
2) Shell Sort 변종: Tokuda gap sequence
3) 병렬 정렬: Multiprocessing 기반 Parallel Merge Sort
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import List, Sequence

def quick_sort_variant(values: Sequence[int]) -> List[int]:
    """
    Median-of-Three + 3-way partition Quick Sort.

    입력 시퀀스를 복사하여 정렬된 새 리스트를 반환한다.
    """

    arr = list(values)
    if len(arr) <= 1:
        return arr

    _quick_sort_recursive(arr, 0, len(arr) - 1)
    return arr


def _quick_sort_recursive(arr: List[int], lo: int, hi: int) -> None:
    """재귀적으로 구간 [lo, hi]를 정렬한다."""
    if lo >= hi:
        return

    pivot = _median_of_three(arr, lo, hi)
    lt, gt = _partition_3way(arr, lo, hi, pivot)

    _quick_sort_recursive(arr, lo, lt - 1)
    _quick_sort_recursive(arr, gt + 1, hi)


def _median_of_three(arr: List[int], lo: int, hi: int) -> int:
    """
    lo, mid, hi 위치의 값 중 중앙값을 피벗 값으로 반환.
    정렬된/역정렬된 입력에서 극단 분할을 완화하려는 목적이다.
    """

    mid = (lo + hi) // 2
    a, b, c = arr[lo], arr[mid], arr[hi]

    if a <= b <= c or c <= b <= a:
        return b
    if b <= a <= c or c <= a <= b:
        return a
    return c


def _partition_3way(arr: List[int], lo: int, hi: int, pivot: int) -> tuple[int, int]:
    """
    Dutch National Flag 방식의 3-way partition.

    반환값 (lt, gt)의 의미:
      - arr[lo:lt] < pivot
      - arr[lt:gt+1] == pivot
      - arr[gt+1:hi+1] > pivot
    """

    lt = lo
    i = lo
    gt = hi

    while i <= gt:
        if arr[i] < pivot:
            arr[lt], arr[i] = arr[i], arr[lt]
            lt += 1
            i += 1
        elif arr[i] > pivot:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
        else:
            i += 1

    return lt, gt


def shell_sort_tokuda(values: Sequence[int]) -> List[int]:
    """
    Tokuda gap sequence 기반 Shell Sort.

    gap 수열을 큰 값부터 적용하며, 각 gap마다 gapped insertion sort 수행.
    """

    arr = list(values)
    n = len(arr)
    if n <= 1:
        return arr

    gaps = _tokuda_gaps(n)
    for gap in gaps:
        for i in range(gap, n):
            current = arr[i]
            j = i

            while j >= gap and arr[j - gap] > current:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = current

    return arr


def _tokuda_gaps(n: int) -> List[int]:
    """
    n보다 작은 Tokuda gap들을 생성하여 내림차순으로 반환.
    마지막에는 반드시 gap=1이 포함되어 완전 정렬이 보장된다.
    """

    gaps: List[int] = []
    k = 1
    while True:
        gap = int(((9**k - 4**k) / (5 * (4 ** (k - 1)))) + 0.9999999)
        if gap > n:
            break
        gaps.append(gap)
        k += 1

    if 1 not in gaps:
        gaps.append(1)

    gaps = sorted(set(gaps), reverse=True)
    return gaps


def parallel_merge_sort(values: Sequence[int], max_workers: int | None = None) -> List[int]:
    """
    Multiprocessing 기반 병렬 병합정렬.

    전략:
    1) 입력을 worker 개수만큼 분할
    2) 각 청크를 병렬로 정렬(sorted 사용)
    3) 정렬된 청크들을 순차 병합
    """

    arr = list(values)
    n = len(arr)
    if n <= 1:
        return arr

    if max_workers is None or max_workers < 1:
        max_workers = 4

    chunk_size = (n + max_workers - 1) // max_workers
    chunks = [arr[i : i + chunk_size] for i in range(0, n, chunk_size)]

    # 각 청크를 독립 프로세스에서 정렬한다.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        sorted_chunks = list(executor.map(sorted, chunks))

    # 정렬된 청크들을 왼쪽부터 반복 병합한다.
    result = sorted_chunks[0]
    for chunk in sorted_chunks[1:]:
        result = _merge_two_sorted_lists(result, chunk)

    return result


def _merge_two_sorted_lists(left: List[int], right: List[int]) -> List[int]:
    """두 정렬 리스트를 병합하여 새 정렬 리스트를 반환."""

    merged: List[int] = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])

    return merged
