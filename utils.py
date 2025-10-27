from typing import List

def counts_to_cumulative_ascending(counts: list[int]) -> list[int]:
    """Return ascending cumulative array: cum[0]=0, cum[n]=total."""
    cum = [0]
    s = 0
    for c in counts:
        s += c
        cum.append(s)
    return cum

def probs_to_counts_largest_remainder(probs: List[float], slots: int) -> List[int]:
        raw = [float(p) for p in probs]
        ssum = sum(raw)
        scaled = [r * slots / ssum for r in raw]
        floors = [int(x) for x in scaled]
        remainder = slots - sum(floors)
        fracs = sorted(((i, scaled[i] - floors[i]) for i in range(len(probs))),
                       key=lambda x: x[1], reverse=True)
        i = 0
        while remainder > 0 and i < len(fracs):
            floors[fracs[i][0]] += 1
            remainder -= 1
            i += 1
        for p, c in zip(raw, floors):
            if p > 0.0 and c == 0:
                raise ValueError("Precision too low")
        return floors

def counts_to_cum_desc(counts: List[int]) -> List[int]:
    total = sum(counts)
    cum = [total]
    s = 0
    for c in counts:
        s += c
        cum.append(total - s)
    return cum

