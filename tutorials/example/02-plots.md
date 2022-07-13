@def title = "Example 02 - Plots
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
 -0.28704486173331617
 -0.26655514132145813
 -0.1418605465484728
 -0.15323550025321422
 -0.28162208660894955
 -0.14387424157319195
 -0.19404813454967815
 -0.0667106495563769
 -0.1253960497277345
 -0.11978737618481351
 -0.0050333172640418915
 -0.019906794908332436
  0.1455829872186409
 -0.04851355974815816
  0.25941529607556635
  0.25762990428276067
  0.26313699145959574
  0.20799052865721543
  0.2766272328401656
  0.27086826958558274
```





```
using JSOTutorials
JSSOTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])
```
