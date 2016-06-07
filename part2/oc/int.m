(* $Assumptions = {Element[{rs,R,x}, Reals],  rs>0 , R>0} *)

f[x_]:= Sqrt[(1+Cos[x])/2 - ((1+Cos[x])/2)^2] + ArcCos[Sqrt[(1+Cos[x])/2]]
Print[f[x]]
g[x_]:= D[f[x],x]
Print[g[x]]
h[x_]:=1/ Sqrt[1-(2 rs)/ (R (1+Cos[x])) ] * g[x]
Print[FullSimplify[h[x]] // TeXForm]
(* l[x_]:=Integrate[FullSimplify[h[x]],x] *)
l[x_]:=Integrate[h[x],x]
 Print[FullSimplify[l[x]]] 

Print[FullSimplify[l[x]] // TeXForm]



