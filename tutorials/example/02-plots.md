@def title = "Example 02 - Plots"
@def showall = true
@def tags = "["example", "plots"]

\preamble{Abel Soares Siqueira}

```julia
# using Plots

x = range(-2, 2, length=20)
y = sin.(x) * 0.2 + randn(20) * 0.1
# plot(
#   x,
#   y,
#   m = (3, :circle),
# )
```

```
20-element Vector{Float64}:
 -0.06473797377748335
 -0.0689825168812345
 -0.30320510742781215
 -0.16046762403920495
 -0.3027920968117236
 -0.21328371288426337
 -0.05346589632296908
 -0.0994484022547249
 -0.24553539021744458
 -0.04596326884600502
 -0.11100918197530935
 -0.00392243573948254
  0.11980060810355125
  0.26540145595374676
  0.14185879327043488
  0.2547799789730892
  0.4147854790038812
  0.14046879022683306
  0.2340659069948743
  0.33278648725693777
```





```
using JSOTutorials
JSSOTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])
```
