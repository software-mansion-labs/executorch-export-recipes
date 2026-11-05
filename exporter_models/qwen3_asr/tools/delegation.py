"""Per-method delegate-call vs host-kernel instruction counts of a .pte (structural delegation check)."""
import sys, collections
from executorch.exir._serialize._program import deserialize_pte_binary
prog = deserialize_pte_binary(open(sys.argv[1], "rb").read())
prog = getattr(prog, "program", prog)
for plan in prog.execution_plan:
    if not plan.chains or not any(ch.instructions for ch in plan.chains): continue
    ops = [o.name + ("." + o.overload if o.overload else "") for o in plan.operators]
    deleg = [d.id for d in plan.delegates]
    host, ndel = collections.Counter(), collections.Counter()
    for ch in plan.chains:
        for ins in ch.instructions:
            a = ins.instr_args; t = type(a).__name__
            if t == "DelegateCall": ndel[deleg[a.delegate_index]] += 1
            elif t == "KernelCall": host[ops[a.op_index]] += 1
    print(f"{plan.name}: delegate calls {dict(ndel)} | host kernels {sum(host.values())}: {dict(host.most_common())}")
