@def title = "Introduction to QuadraticModels"
@def showall = true
@def tags = ["models"]

\preamble{Geoffroy Leconte}



In this tutorial you will learn how to create and use QuadraticModels.

\toc

## Create a QuadraticModel

QuadraticModels represent the optimization problem

$$
\begin{aligned}
\min \quad & \tfrac{1}{2}x^T H x + c^T x + c0 \\
& lcon  \leq Ax \leq ucon \\
& \ell \leq  x \leq u.
\end{aligned}
$$

`H` should be a lower triangular matrix.
QuadraticModels can work with different input matrix types:

```julia
H = [
  6.0 2.0 1.0
  2.0 5.0 2.0
  1.0 2.0 4.0
]
c = [-8.0; -3; -3]
A = [
  1.0 0.0 1.0
  0.0 2.0 1.0
]
lcon = [-2.0; 0.]
ucon = [10.0; 20.0]
l = [0.0; 0; 0]
u = [Inf; Inf; Inf]
using LinearAlgebra, SparseArrays, QuadraticModels, SparseMatricesCOO, LinearOperators
HCOO = SparseMatrixCOO(tril(H))
ACOO = SparseMatrixCOO(A)
HCSC = sparse(tril(H))
ACSC = sparse(A)
qmCSC = QuadraticModel(c, HCSC, A = ACSC, lcon = lcon, ucon = ucon, lvar = l, uvar = u,
                       c0 = 0.0, name = "QM_CSC")
```

```
Error: ArgumentError: Package QuadraticModels not found in current path:
- Run `import Pkg; Pkg.add("QuadraticModels")` to install the QuadraticMode
ls package.
```



```julia
qmCOO = QuadraticModel(c, HCOO, A = ACOO, lcon = lcon, ucon = ucon, lvar = l, uvar = u,
                       c0 = 0.0, name = "QM_COO")
```

```
Error: UndefVarError: ACOO not defined
```



```julia
qmDense = QuadraticModel(c, tril(H), A = A, lcon = lcon, ucon = ucon, lvar = l, uvar = u,
                         c0 = 0.0, name = "QM_Dense")
```

```
Error: UndefVarError: QuadraticModel not defined
```



```julia
qmLinop = QuadraticModel(c, LinearOperator(Symmetric(H)), A = LinearOperator(A),
                         lcon = lcon, ucon = ucon, lvar = l, uvar = u,
                         c0 = 0.0, name = "QM_Linop")
```

```
Error: UndefVarError: LinearOperator not defined
```





You can also create a COO QuadraticModel directly from the coordinates (without using [`SparseMatricesCOO.jl`](https://github.com/JuliaSmoothOptimizers/SparseMatricesCOO.jl)):

```julia
Hrows, Hcols, Hvals = findnz(sparse(tril(H)))
Arows, Acols, Avals = findnz(sparse(A))
qmCOO2 = QuadraticModel(c, Hrows, Hcols, Hvals, Arows = Arows, Acols = Acols, Avals = Avals,
                        lcon = lcon, ucon = ucon, lvar = l, uvar = u,
                        c0 = 0.0, name = "QM_COO2")
```

```
Error: UndefVarError: QuadraticModel not defined
```





## Convert your model

Some functions work best with SparseMatricesCOO.
You can convert your QuadraticModel with

```julia
T = Float64
S = Vector{T}
qmCOO3 = convert(QuadraticModel{T, S, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}},
                 qmDense)

qmCOO4 = convert(QuadraticModel{T, S, SparseMatrixCOO{T, Int}, SparseMatrixCOO{T, Int}},
                 qmCSC)
```

```
Error: UndefVarError: SparseMatrixCOO not defined
```





## Use the NLPModels.jl API

You can use the API from [`NLPModels.jl`](https://juliasmoothoptimizers.github.io/NLPModels.jl/stable/api/#Reference-guide) with QuadraticModels.
Here are some examples:

```julia
using NLPModels
x = rand(3)
grad(qmCOO, x)
```

```
Error: ArgumentError: Package NLPModels not found in current path:
- Run `import Pkg; Pkg.add("NLPModels")` to install the NLPModels package.
```



```julia
hess(qmCOO, x)
```

```
Error: UndefVarError: hess not defined
```





It is possible to convert the model to a QuadraticModel with linear inequality constraints to equality constraints and bounds using [`SlackModel`](https://juliasmoothoptimizers.github.io/NLPModelsModifiers.jl/stable/reference/#NLPModelsModifiers.SlackModel)

```julia
using NLPModelsModifiers
qmSlack = SlackModel(qmCOO)
```

```
Error: ArgumentError: Package NLPModelsModifiers not found in current path:
- Run `import Pkg; Pkg.add("NLPModelsModifiers")` to install the NLPModelsM
odifiers package.
```





## Read MPS/SIF files

You can read directly MPS or SIF files using [`QPSReader.jl`](https://github.com/JuliaSmoothOptimizers/QPSReader.jl)

```julia
using QPSReader
qps = readqps("AFIRO.SIF")
qmCOO4 = QuadraticModel(qps)
```

```
Error: ArgumentError: Package QPSReader not found in current path:
- Run `import Pkg; Pkg.add("QPSReader")` to install the QPSReader package.
```





## Solving

You can use [`RipQP.jl`](https://github.com/JuliaSmoothOptimizers/RipQP.jl) to solve QuadraticModels:

```julia
using RipQP
stats = ripqp(qmCOO)
println(stats)
```

```
Error: ArgumentError: Package RipQP not found in current path:
- Run `import Pkg; Pkg.add("RipQP")` to install the RipQP package.
```


