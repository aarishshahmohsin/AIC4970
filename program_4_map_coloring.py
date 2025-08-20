from collections import deque
from typing import Dict, List, Set, Tuple, Optional
import time

class CSP:
    """
    Constraint Satisfaction Problem (CSP) class.

    variables: list of variable names.
    domains: dictionary mapping variable -> list of possible values.
    neighbors: dictionary mapping variable -> set of neighboring variables.
    """
    def __init__(self, variables: List[str], domains: Dict[str, List[str]], neighbors: Dict[str, Set[str]]):
        self.variables = list(variables)
        self.domains = {v: list(domains[v]) for v in self.variables}
        self.neighbors = {v: set(neighbors.get(v, set())) for v in self.variables}
        self.n_assigns = 0
        self.n_backtracks = 0

    def consistent(self, var: str, value: str, assignment: Dict[str, str]) -> bool:
        """
        Consistency check: ensures that assigning var=value is consistent
        with current assignment (neighbors must not have the same value).
        """
        for n in self.neighbors[var]:
            if assignment.get(n) == value:
                return False
        return True

    def assign(self, var: str, value: str, assignment: Dict[str, str]):
        """Assigns value to variable and increments assignment counter."""
        assignment[var] = value
        self.n_assigns += 1

    def unassign(self, var: str, assignment: Dict[str, str]):
        """Removes a variable assignment if it exists."""
        if var in assignment:
            del assignment[var]


'''
Minimum Remaining Value (MRV) and Degree Heuristic

- MRV: choose the unassigned variable with the fewest available legal values (smallest domain).
- Degree Heuristic: If tie in MRV, choose variable with the largest number of unassigned neighbors.
'''

def select_unassigned_variable(csp: CSP, assignment: Dict[str, str], local_domains: Dict[str, List[str]], heuristic: str):
    unassigned = [v for v in csp.variables if v not in assignment]
    if heuristic == 'basic':
        return unassigned[0]
    m = min(len(local_domains[v]) for v in unassigned)
    candidates = [v for v in unassigned if len(local_domains[v]) == m]
    if heuristic == 'mrv+deg' and len(candidates) > 1:
        def degree_unassigned(v):
            return sum(1 for n in csp.neighbors[v] if n not in assignment)
        maxdeg = max(degree_unassigned(v) for v in candidates)
        candidates = [v for v in candidates if degree_unassigned(v) == maxdeg]
    return candidates[0]


'''
Least Constraining Value (LCV)

LCV orders values so that the chosen value eliminates the fewest choices
in the domains of neighbor variables.
'''

def order_domain_values(csp: CSP, var: str, assignment: Dict[str, str], local_domains: Dict[str, List[str]], heuristic: str):
    if heuristic != 'lcv':
        return list(local_domains[var])
    counts = []
    for val in local_domains[var]:
        eliminated = 0
        for n in csp.neighbors[var]:
            if n not in assignment:
                eliminated += sum(1 for v in local_domains[n] if v == val)
        counts.append((eliminated, val))
    counts.sort()
    return [val for _, val in counts]


'''
Forward Checking (FC)

After assigning var=value, forward checking removes this value from each
unassigned neighbor's domain. If any neighbor's domain becomes empty,
then failure is detected immediately.
'''

def forward_checking(csp: CSP, var: str, value: str, assignment: Dict[str, str], local_domains: Dict[str, List[str]], logger=None, depth=0):
    pruned = []
    if logger:
        logger(f"FC: checking neighbors of {var} after assigning {value}", depth)
    for n in csp.neighbors[var]:
        if n in assignment:
            continue
        if value in local_domains[n]:
            local_domains[n].remove(value)
            pruned.append((n, value))
            if logger:
                logger(f"FC: pruned value {value} from domain of {n}; new domain: {local_domains[n]}", depth+1)
            if not local_domains[n]:
                if logger:
                    logger(f"FC: failure, domain of {n} is empty after pruning", depth+1)
                return False, pruned
    return True, pruned


'''
Arc Consistency (AC-3)

AC-3 enforces arc consistency. For binary constraints, this requires that
for every value in a variable's domain, there is some compatible value in
the domain of each neighbor. For inequality constraints, revise removes
values that have no different value in the neighbor's domain.
'''

def AC3(csp: CSP, local_domains: Dict[str, List[str]], arcs: Optional[List[Tuple[str, str]]] = None, logger=None, depth=0):
    queue = deque()
    if arcs is None:
        for Xi in csp.variables:
            for Xj in csp.neighbors[Xi]:
                queue.append((Xi, Xj))
    else:
        for arc in arcs:
            queue.append(arc)
    pruned = []

    def revise(Xi, Xj):
        revised = False
        to_remove = []
        for x in list(local_domains[Xi]):
            if not any(x != y for y in local_domains[Xj]):
                to_remove.append(x)
        for x in to_remove:
            local_domains[Xi].remove(x)
            pruned.append((Xi, x))
            revised = True
            if logger:
                logger(f"AC3: removed value {x} from domain of {Xi} because no supporting value in {Xj}", depth+1)
        return revised

    if logger:
        logger(f"AC3: starting with {len(queue)} arcs in queue", depth)
    while queue:
        Xi, Xj = queue.popleft()
        if logger:
            logger(f"AC3: processing arc ({Xi},{Xj}); domains: {Xi}:{local_domains[Xi]}, {Xj}:{local_domains[Xj]}", depth+1)
        if revise(Xi, Xj):
            if not local_domains[Xi]:
                if logger:
                    logger(f"AC3: failure, domain of {Xi} empty after revise", depth+1)
                return False, pruned
            for Xk in csp.neighbors[Xi] - {Xj}:
                queue.append((Xk, Xi))
                if logger:
                    logger(f"AC3: queued arc ({Xk},{Xi}) due to change in {Xi}", depth+2)
    if logger:
        logger(f"AC3: finished; total pruned: {len(pruned)}", depth)
    return True, pruned


'''
Backtracking Search

inference = {'none', 'fc', 'ac3'}
variable heuristics = {'basic', 'mrv', 'mrv+deg'}
value heuristics = {'basic', 'lcv'}
use_ac3_at_start: run AC3 before searching.
'''

def backtracking_search(csp: CSP,
                        inference: str = 'none',
                        var_heuristic: str = 'basic',
                        val_heuristic: str = 'basic',
                        use_ac3_at_start: bool = False,
                        time_limit: float = 10.0,
                        verbose: bool = True,
                        step_delay: float = 0.0):
    """
    Backtracking search with verbose step-by-step tracing.

    - verbose: if True prints each step.
    - step_delay: optional delay (seconds) between printed steps to watch progress; default 0 (no delay).
    """
    start = time.time()
    assignment: Dict[str, str] = {}
    csp.n_assigns = 0
    csp.n_backtracks = 0
    local_domains = {v: list(csp.domains[v]) for v in csp.variables}

    def logger(msg: str, depth: int = 0):
        if not verbose:
            return
        indent = '  ' * depth
        elapsed = time.time() - start
        print(f"[{elapsed:6.3f}s] STEP {logger.step_count:04d}: {indent}{msg}")
        logger.step_count += 1
        if step_delay:
            time.sleep(step_delay)
    logger.step_count = 1

    def print_graph_state(depth=0):
        logger("Current CSP graph (neighbors):", depth)
        for v in csp.variables:
            logger(f"  {v}: neighbors -> {sorted(list(csp.neighbors[v]))}", depth+1)
        logger("Current domains:", depth)
        for v in csp.variables:
            dom = local_domains[v]
            assigned = assignment.get(v)
            logger(f"  {v}: domain={dom} {'(assigned='+assigned+')' if assigned else ''}", depth+1)

    initial_pruned = []
    logger("Initial graph and domains:", 0)
    print_graph_state(0)

    if use_ac3_at_start:
        logger("Running AC3 at start to prune domains...", 0)
        ok, pruned0 = AC3(csp, local_domains, logger=logger, depth=1)
        initial_pruned = pruned0
        logger(f"AC3 initial pruning removed {len(initial_pruned)} values: {initial_pruned}", 0)
        if not ok:
            logger("AC3 at start failed: no solution.", 0)
            return None, {
                "success": False,
                "time_s": time.time() - start,
                "assigns": csp.n_assigns,
                "backtracks": csp.n_backtracks,
                "initial_pruned": len(initial_pruned)
            }

    def backtrack(depth=0):
        logger(f"Backtrack called (depth={depth}). Assignment size={len(assignment)}", depth)
        if len(assignment) == len(csp.variables):
            logger("All variables assigned -> solution found.", depth)
            return dict(assignment)

        print_graph_state(depth)

        var = select_unassigned_variable(csp, assignment, local_domains, var_heuristic)
        logger(f"Selected variable '{var}' using heuristic '{var_heuristic}' (domain={local_domains[var]})", depth)

        for value in order_domain_values(csp, var, assignment, local_domains, val_heuristic):
            logger(f"Trying {var} = {value} (value-ordering: {val_heuristic})", depth+1)
            if not csp.consistent(var, value, assignment):
                logger(f"Inconsistent: neighbor already has value {value}. Skipping.", depth+2)
                continue
            csp.assign(var, value, assignment)
            logger(f"Assigned {var} = {value}. Assigns count: {csp.n_assigns}", depth+2)
            pruned = []
            ok = True
            if inference == 'fc':
                ok, pruned = forward_checking(csp, var, value, assignment, local_domains, logger=logger, depth=depth+2)
            elif inference == 'ac3':
                arcs = [(n, var) for n in csp.neighbors[var]]
                logger(f"Running AC3 after assigning {var} (arcs={arcs})", depth+2)
                ok, pruned = AC3(csp, local_domains, arcs=arcs, logger=logger, depth=depth+3)

            if ok:
                result = backtrack(depth+1)
                if result is not None:
                    return result

            # undo
            csp.unassign(var, assignment)
            logger(f"Unassigned {var}. Restoring pruned values (if any).", depth+2)
            for (v, val) in pruned:
                if val not in local_domains[v]:
                    local_domains[v].append(val)
                    logger(f"Restored value {val} to domain of {v}; domain now: {local_domains[v]}", depth+3)
            logger(f"Backtracking from {var}={value}", depth+2)
        csp.n_backtracks += 1
        logger(f"No values left for variable {var} -> increment backtracks to {csp.n_backtracks}", depth)
        return None

    sol = backtrack()
    metrics = {
        "success": sol is not None,
        "time_s": time.time() - start,
        "assigns": csp.n_assigns,
        "backtracks": csp.n_backtracks,
        "initial_pruned": len(initial_pruned)
    }
    logger(f"Search finished. Success={metrics['success']}. Time: {metrics['time_s']:.3f}s Assigns: {metrics['assigns']} Backtracks: {metrics['backtracks']}", 0)
    return sol, metrics


def australia_graph():
    regions = ["WA","NT","SA","Q","NSW","V","T"]
    edges = {
        ("WA","NT"),("WA","SA"),("NT","SA"),("NT","Q"),
        ("SA","Q"),("SA","NSW"),("SA","V"),
        ("Q","NSW"),("NSW","V")
    }
    neighbors = {r:set() for r in regions}
    for a,b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    return regions, neighbors


def square_cross_graph():
    regions = ["A","B","C","D","E"]
    edges = {
        ("A","B"),("B","C"),("C","D"),("D","A"),
        ("E","A"),("E","B"),("E","C"),("E","D")
    }
    neighbors = {r:set() for r in regions}
    for a,b in edges:
        neighbors[a].add(b); neighbors[b].add(a)
    return regions, neighbors

# Demo when run as main
if __name__ == '__main__':
    regions, neighbors = australia_graph()
    colors = ["Red","Green","Blue","Yellow"]
    domains = {r:list(colors) for r in regions}
    csp = CSP(regions, domains, neighbors)

    # change settings here
    sol, metrics = backtracking_search(csp,
                                      inference='fc',        # 'none', 'fc', or 'ac3'
                                      var_heuristic='mrv+deg',
                                      val_heuristic='lcv',
                                      use_ac3_at_start=True,
                                      verbose=True,
                                      step_delay=0.0)  # set step_delay>0 to slow the printed steps
    print("\nFinal Solution:", sol)
    print("Metrics:", metrics)