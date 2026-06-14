# PTX And SASS Evidence

PTX and SASS are how I check what the compiler actually did.

## PTX

PTX is virtual ISA.

Useful for:

- seeing target-gated instructions
- checking address spaces
- checking approximate lowering
- learning NVIDIA's abstraction model

But PTX is not the final machine code.

## SASS

SASS is the real machine instruction stream.

Useful for:

- checking actual load/store instructions
- seeing Tensor Core instructions
- seeing instruction mix
- comparing targets
- verifying whether a compiler optimization happened

## Commands

```bash
nvcc -ptx kernel.cu -o kernel.ptx
nvcc kernel.cu -arch=sm_89 -o kernel
cuobjdump --dump-sass kernel
```

Change `sm_89` to the actual GPU target.

## Checklist

- [ ] generate PTX for one simple kernel
- [ ] generate SASS for one simple kernel
- [ ] identify load/store instructions
- [ ] identify arithmetic instructions
- [ ] identify branch/predicate instructions
- [ ] compare SASS across two `sm_XX` targets if toolkit supports it
- [ ] connect one profiler bottleneck to one disassembly fact

