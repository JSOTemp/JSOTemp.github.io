@def title = "Example 02 - Plots"
@def showall = true
@def tags = ["example", "plots"]

\preamble{Abel Soares Siqueira}

```julia
# using Plots

x = range(-2, 2, length=20)
y = sin.(x) CONTRIBUTING.md Manifest.toml Project.toml README.md auto-build.sh build-site.jl jso-banner.png jso.png markdown parsed src test tutorials 0.2 + randn(20) CONTRIBUTING.md Manifest.toml Project.toml README.md auto-build.sh build-site.jl jso-banner.png jso.png markdown parsed src test tutorials 0.1
# plot(
# x,
# y,
# m = (3, :circle),
# )
```

```
20-element Vector{Float64}:
0.012798046082130698
0.02251544429672181
-0.13871576356795068
-0.30175277434194014
-0.2973206531804806
-0.2575469915620945
-0.20176439837725463
-0.2402969079558519
-0.1611239257718413
0.15988538812384756
0.14028726776740055
0.10107956623541389
0.13360163437392214
0.09363218462571055
0.2647503400675581
0.32225082963198337
0.17156497664668802
0.21394498828162203
0.3349821092631337
0.1857976650769929
```





```
using JSOTutorials
JSSOTutorials.tutorial_footer(WEAVE_ARGS[:folder],WEAVE_ARGS[:file])
```
