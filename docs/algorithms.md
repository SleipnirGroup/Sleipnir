# Algorithms

## Reverse accumulation automatic differentiation

In reverse accumulation AD, the dependent variable to be differentiated is fixed and the derivative is computed with respect to each subexpression recursively. In a pen-and-paper calculation, the derivative of the outer functions is repeatedly substituted in the chain rule:

(∂y/∂x) = (∂y/∂w₁) ⋅ (∂w₁/∂x) = ((∂y/∂w₂) ⋅ (∂w₂/∂w₁)) ⋅ (∂w₁/∂x) = ...

In reverse accumulation, the quantity of interest is the adjoint, denoted with a bar (w̄); it is a derivative of a chosen dependent variable with respect to a subexpression w: ∂y/∂w.

Given the expression f(x₁,x₂)=sin(x₁) + x₁x₂, the computational graph is:
```
               f(x₁,x₂)
                  |
                  w₅     w₅=w₄+w₃
                 /  \
                /    \
 w₄=sin(w₁)    w₄     w₃    w₃=w₁w₂
               |   /  |
     w₁=x₁     w₁     w₂    w₂=x₃
```

The operations to compute the derivative:

w̄₅ = 1 (seed)\
w̄₄ = w̄₅(∂w₅/∂w₄) = w̄₅\
w̄₃ = w̄₅(∂w₅/∂w₃) = w̄₅\
w̄₂ = w̄₃(∂w₃/∂w₂) = w̄₃w₁\
w̄₁ = w̄₄(∂w₄/∂w₁) + w̄₃(∂w₃/∂w₁) = w̄₄cos(w₁) + w̄₃w₂

https://en.wikipedia.org/wiki/Automatic_differentiation#Beyond_forward_and_reverse_accumulation
