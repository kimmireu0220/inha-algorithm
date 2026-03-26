"""
EEC3414 알고리즘 H/W #1 실험 실행 스크립트

기능:
1) n 크기별 랜덤 데이터셋 생성(동일 시드 재현)
2) 세 정렬 알고리즘 실행시간 측정
3) CSV 결과 저장 + 그래프 저장
"""

from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Callable, Dict, List

import matplotlib.pyplot as plt

from src.sorting_algorithms import (
    parallel_merge_sort,
    quick_sort_variant,
    shell_sort_tokuda,
)


# 과제 예시 n + 확장 n
DATASET_SIZES = [10_000, 50_000, 100_000, 500_000, 1_000_000]
RANDOM_SEED = 3414
REPEAT_PER_CASE = 3
MAX_WORKERS = 8


def generate_dataset(n: int, rng: random.Random) -> List[int]:
    """0~10^9 범위의 정수를 n개 생성한다."""
    return [rng.randint(0, 1_000_000_000) for _ in range(n)]


def measure_time(sort_func: Callable[[List[int]], List[int]], data: List[int]) -> float:
    """
    정렬 함수 실행 시간을 초 단위로 반환.
    결과가 실제 정렬되었는지도 검증한다.
    """
    start = time.perf_counter()
    result = sort_func(data)
    elapsed = time.perf_counter() - start

    if result != sorted(data):
        raise ValueError(f"{sort_func.__name__} 결과가 올바르게 정렬되지 않았습니다.")

    return elapsed


def run() -> None:
    project_root = Path(__file__).resolve().parent
    result_dir = project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(RANDOM_SEED)
    rows: List[Dict[str, float | int | str]] = []

    for n in DATASET_SIZES:
        print(f"데이터 생성 중: n={n}")
        base_data = generate_dataset(n, rng)

        cases: Dict[str, Callable[[List[int]], List[int]]] = {
            "quick_median3_3way": quick_sort_variant,
            "shell_tokuda": shell_sort_tokuda,
            "parallel_merge": lambda arr: parallel_merge_sort(arr, max_workers=MAX_WORKERS),
        }

        for algo_name, algo_fn in cases.items():
            trial_times = []
            for trial in range(1, REPEAT_PER_CASE + 1):
                # 동일 데이터셋 비교를 위해 반드시 같은 입력을 복사 사용
                data = base_data.copy()
                elapsed = measure_time(algo_fn, data)
                trial_times.append(elapsed)
                print(f"  {algo_name:20s} trial#{trial}: {elapsed:.6f} sec")

            avg_time = sum(trial_times) / len(trial_times)
            rows.append(
                {
                    "n": n,
                    "algorithm": algo_name,
                    "avg_time_sec": avg_time,
                    "min_time_sec": min(trial_times),
                    "max_time_sec": max(trial_times),
                }
            )
            print(f"  -> 평균: {avg_time:.6f} sec")
        print("-" * 60)

    csv_path = result_dir / "benchmark_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["n", "algorithm", "avg_time_sec", "min_time_sec", "max_time_sec"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV 저장 완료: {csv_path}")
    plot_results(rows, result_dir / "performance_plot.png")


def plot_results(rows: List[Dict[str, float | int | str]], output_path: Path) -> None:
    """n 대비 평균 실행시간 그래프를 저장한다."""
    grouped: Dict[str, List[tuple[int, float]]] = {}
    for row in rows:
        algo = str(row["algorithm"])
        grouped.setdefault(algo, []).append((int(row["n"]), float(row["avg_time_sec"])))

    plt.figure(figsize=(10, 6))
    for algo, points in grouped.items():
        points.sort(key=lambda x: x[0])
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        plt.plot(x, y, marker="o", label=algo)

    plt.title("Sorting Performance Comparison")
    plt.xlabel("Input Size (n)")
    plt.ylabel("Average Execution Time (sec)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"그래프 저장 완료: {output_path}")


if __name__ == "__main__":
    run()
