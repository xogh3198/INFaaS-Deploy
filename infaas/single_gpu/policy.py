"""Single-device reduction of INFaaS's existing memory/loading cost objective.

This is an explicit adapter, NOT the full original distributed autoscaler.
Capacity/SLO are constraints; accuracy is an admission constraint, not a new
objective. Never switch just to make a workload demonstrate a model change.
"""
from dataclasses import dataclass
import math

from infaas.policy.costs import action_cost, instance_cost
from infaas.policy.types import VariantProfile


@dataclass(frozen=True)
class Candidate:
    model: str
    hardware: str
    batch: int
    accuracy: float
    p50_ms: float
    p95_ms: float
    capacity_qps: float
    load_ms: float
    memory_bytes: int

    def profile(self):
        return VariantProfile(f"{self.model}__{self.hardware}", self.model,
            self.hardware, self.p50_ms, self.load_ms, self.capacity_qps, self.memory_bytes)

    def as_dict(self):
        return {**vars(self), "instance_cost": instance_cost(self.profile())}


def candidates(profiles, entries, hardware, slo_ms, min_accuracy=0.0):
    result = []
    for model, profile in profiles.items():
        # QuerySLO.MinAccuracy is protobuf float32; tolerate wire-rounding of
        # supplied decimal scores (e.g. 70.93 becomes 70.9300003).
        if model not in entries or entries[model]["accuracy"] + 1e-4 < min_accuracy:
            continue
        for batch, row in profile["batches"].items():
            # Common non-preemptive execution budget: leave SLO/2 for queueing.
            # CPU preprocessing is already complete at the ready queue, but its
            # elapsed time is NEVER removed from an individual request deadline.
            if row["p50_ms"] >= slo_ms / 2:
                continue
            result.append(Candidate(model, hardware, int(batch), entries[model]["accuracy"],
                row["p50_ms"], row["p95_ms"], row["capacity_qps"],
                profile["load_ms"], row["memory_bytes"]))
    return result


def choose(options, demand_qps, current=None, slack=1.05):
    if not options:
        raise ValueError("No candidate meets accuracy and profiled SLO/2 constraints")
    if not math.isfinite(demand_qps) or demand_qps < 0 or slack < 1:
        raise ValueError("Finite nonnegative demand and slack >= 1 required")
    if any(not math.isfinite(instance_cost(row.profile())) for row in options):
        raise ValueError("HW_COST must contain a finite price for the selected GPU type")
    feasible = [row for row in options if row.capacity_qps >= demand_qps * slack]
    pool = feasible or options

    def score(row):
        # A batch adjustment on an already-resident model does not reload weights.
        value = instance_cost(row.profile())
        if current is None or current.model != row.model:
            # C_current + Cost(+1 new) + Cost(-1 current) = Cost(+1 new).
            # Thus the old running-cost term cancels on a single-device swap.
            value = action_cost(row.profile(), 1)
        # Preserve cost priority. Accuracy only resolves a remaining exact tie.
        return (value, -row.accuracy,
                row.batch, row.p50_ms, row.model)

    # Under infeasible overload, maximize available capacity instead of pretending
    # an unlimited number of replicas can be placed on this single device.
    selected = min(pool, key=score) if feasible else min(pool, key=lambda row: (-row.capacity_qps, score(row)))
    return selected, {
        "demand_qps": demand_qps, "required_qps_with_slack": demand_qps * slack,
        "capacity_sufficient": bool(feasible), "objective": "infaas-memory-loading-cost",
        "reason": "minimum_cost_feasible" if feasible else "overload_max_capacity",
        "candidate_count": len(options), "feasible_count": len(feasible),
        "score": score(selected)[0],
    }
