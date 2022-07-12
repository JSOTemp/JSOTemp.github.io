---
author: "Abel Soares Siqueira"
title: "Example 01 - No pkgs"
---


This is a simple example.


$$f(x) = x^2 - 5x + 6$$

```julia
f(x) = x^2 - 5x + 6
f.(0.0:0.5:4.0)
```

```
9-element Vector{Float64}:
  6.0
  3.75
  2.0
  0.75
  0.0
 -0.25
  0.0
  0.75
  2.0
```


