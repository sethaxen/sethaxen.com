+++
title = "Differentiating the LU decomposition"
tags = ["linear algebra", "LU decomposition", "automatic differentiation"]
date = Date(2021, 02, 03)
description = "Deriving differentiation rules for the LU decomposition of square and non-square matrices."
published = date
rss_pubdate = date
rss_description = description
maxtoclevel = 2
+++

_I was implementing differentiation rules of the LU decomposition in [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) and needed rules that supported non-square matrices, so I worked them out._

\tableofcontents

## Introduction

For a square matrix $A \in \mathbb{C}^{m \times n}$, the [LU decomposition](https://en.wikipedia.org/wiki/LU_decomposition) is useful for solving systems of equations, inverting $A$, and computing the determinant.

This post assumes familiarity with terminology of forward- and reverse-mode differentiation rules.
For a succinct review, I recommend reading [the ChainRules guide on deriving array rules](https://www.juliadiff.org/ChainRulesCore.jl/stable/arrays.html).

### Definition

The LU decomposition of $A$ is
$$P A = L U.$$
Where $q = \min(m, n)$, $L \in \mathbb{C}^{m \times q}$ is a unit lower triangular matrix, that is, a lower triangular matrix whose diagonal entries are all ones.
$U \in \mathbb{C}^{q \times n}$ is an upper triangular matrix.
$P \in \mathbb{R}^{m \times m}$ is a [permutation matrix](https://en.wikipedia.org/wiki/Permutation_matrix) whose action on a matrix $X$ (i.e. $P X$) reorders the rows of $X$.
As a permutation matrix, $P P^\mathrm{T} = P^\mathrm{T} P = I$.

### Uses

If we have a system of the form of $A X = B$ and would like to solve for $X$, we can use the LU decomposition to write $L U X = P B$.
Then $X = L^{-1} (U^{-1} (P B))$.
Note that the row-swapping action $P B$ can be computed in-place.
We can also easily compute the left-division by the triangular matrices in-place using [forward and back substitution](https://en.wikipedia.org/wiki/Triangular_matrix#Forward_and_back_substitution).

By setting $B$ in the above equation to be the identity matrix, then we can compute the inverse of $A$, that is $X = A^{-1}$.

The determinant of $A$ is $\det(A) = \det(P) \det(L) \det(U)$.
The determinant of a triangular matrix is the product of its diagonal entries, so $\det(L) = 1$, and $\det(U) = \prod_{i=1}^n U_{ii}$.
$\det(P) = (-1)^s$, where $s$ is the number of row swaps encoded by the permutation matrix.
So $\det(A) = (-1)^s \prod_{i=1}^n U_{ii}$, which is very cheap to compute from the decomposition.

The LU decomposition can still be computed when $A$ is wide ($m < n$) or tall ($m > n$).
However, none of the above applications make sense in this case, and I don't know what it's useful for.

### Motivation

Often the LU decomposition is computed using LAPACK subroutines, which cannot be automatically differentiated through.
Hence, it is necessary to implement custom automatic differentiation rules.
[^HoogAnderssenLukas2011] derived a pushforward (AKA forward-mode differentiation rule or Jacobian-vector-product) for the LU decomposition for the square case, but I couldn't find a rule for the wide or tall cases.
This is a problem, because I wanted to implement a generic rule for ChainRules.jl that would work for all dense matrices in Julia, where square and non-square dense matrices are all implemented using the same type `Matrix` (more specifically, as the union of types `StridedMatrix`).
It is not ideal to write a custom rule that will work for only a subset of matrices of a given type.

We could always pad $A$ with zeros to make it square.
However, if $m \ll n$ or $m \gg n$, then this is wasteful.
JAX seems to [use this approach](https://github.com/google/jax/blob/jax-v0.2.9/jax/_src/lax/linalg.py#L826-L871), though it's possible that internally it does something fancier and doesn't explicitly allocate or operate on the padding.

Thankfully, it's not too hard to work out the pushforwards and pullbacks for the non-square case by splitting the matrices into blocks and working out the rules for the individual blocks.
So in this post, we'll review the pushforward for square $A$, also working out its pullback, and then we'll do the same for wide and tall $A$.

## Square $A$

A pushforward for the LU decomposition for square $A$ is already known [^HoogAnderssenLukas2011].
For completeness, I've included its derivation, as well as one for the corresponding pullback.

### Pushforward

We start by differentiating the defining equation:
$$P \dot{A} = \dot{L} U + L \dot{U}$$
We can solve both sides to get
$$L^{-1} P \dot{A} U^{-1} = L^{-1} \dot{L} + \dot{U} U^{-1}$$
$\dot{L}$ and $\dot{U}$ must be at least as sparse as $L$ and $U$, respectively.
Hence, $\dot{U}$ is upper triangular, and because the diagonal of $L$ is constrained to be unit, $\dot{L}$ will be lower triangular with a diagonal of zeros (strictly lower triangular).
Note also that the inverse of a lower/upper triangular matrix is still lower/upper triangular.
Likewise, the product of two lower/upper triangular matrices is still lower/upper triangular.

Hence the right-hand side is the sum of a strictly lower triangular matrix $L^{-1} \dot{L}$ and an upper triangular matrix $\dot{U} U^{-1}$.
Let's introduce the triangularizing operators $\operatorname{triu}(X)$, which extracts the upper triangle of the matrix $X$, and $\operatorname{tril}_-(X)$, which extracts its strict lower triangle (so that $X = \operatorname{tril}_-(X) + \operatorname{triu}(X)$).

Introducing an intermediate $\dot{F}$, we can then solve for $\dot{L}$ and $\dot{U}$:
$$\begin{aligned}
    \dot{F} &= L^{-1} P \dot{A} U^{-1}\\
    \dot{L} &= L \operatorname{tril}_-(\dot{F})\\
    \dot{U} &= \operatorname{triu}(\dot{F}) U
\end{aligned}$$

### Pullback

The corresponding pullback is

$$\begin{aligned}
\overline{F} &= \operatorname{tril}_-(L^\mathrm{H} \overline{L}) + \operatorname{triu}(\overline{U} U^\mathrm{H})\\
\overline{A} &= P^\mathrm{T} L^{-\mathrm{H}} \overline{F} U^{-\mathrm{H}}
\end{aligned}$$

\details{
To find the pullback, we use the identity of reverse-mode differentiation and properties of the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) as described in [the ChainRules guide on deriving array rules](https://www.juliadiff.org/ChainRulesCore.jl/stable/arrays.html).

Here the identity takes the form
$$\Re\ip{\overline{L}}{\dot{L}} + \Re\ip{\overline{U}}{\dot{U}} = \Re\ip{\overline{A}}{\dot{A}}.$$
We want to solve for $\overline{A}$, and we do so by first plugging $\dot{U}$ and $\dot{L}$ into the left-hand side of this identity, manipulating to look like the right-hand side, and then solving for $\overline{A}$.

$$\begin{aligned}
\ip{\overline{L}}{\dot{L}} &= \ip{\overline{L}}{L \operatorname{tril}_-(\dot{F})} = \ip{L^\mathrm{H} \overline{L}}{\operatorname{tril}_-(\dot{F})}\\
\ip{\overline{U}}{\dot{U}} &= \ip{\overline{U}}{\operatorname{triu}(\dot{F}) U} = \ip{\overline{U} U^\mathrm{H}}{\operatorname{triu}(\dot{F})}\\
\end{aligned}$$

Because the Frobenius inner product is the sum of all elements of the element-wise product of the second argument and the complex conjugate of the first argument, then for upper triangular $U$, we have
$$\ip{X}{U} = \sum_{ij} X_{ij}^* U_{ij} = \sum_{ij} \operatorname{triu}(X)_{ij}^* U_{ij} = \ip{\operatorname{triu}(X)}{U}.$$
The same is true for lower-triangular matrices (or, analogously, for any sparsity pattern).
Therefore, 
$$\begin{aligned}
\ip{\overline{L}}{\dot{L}} &= \ip{\operatorname{tril}_-(L^\mathrm{H} \overline{L})}{\dot{F}}\\
\ip{\overline{U}}{\dot{U}} &= \ip{\operatorname{triu}(\overline{U} U^\mathrm{H})}{\dot{F}}\\
\ip{\overline{L}}{\dot{L}} + \ip{\overline{U}}{\dot{U}} &= \ip{\operatorname{tril}_-(L^\mathrm{H} \overline{L}) + \operatorname{triu}(\overline{U} U^\mathrm{H})}{\dot{F}} \doteq \ip{\overline{F}}{\dot{F}},
\end{aligned}$$
where we have introduced an intermediate $\overline{F}$.

Continuing by plugging in $\dot{F}$, we find
$$\ip{\overline{F}}{\dot{F}} = \ip{\overline{F}}{L^{-1} P \dot{A} U^{-1}} = \ip{P^\mathrm{T} L^{-\mathrm{H}} \overline{F} U^{-\mathrm{H}}}{\dot{A}}$$ 

So the pullback of the LU decomposition is written
$$\begin{aligned}
\overline{F} &= \operatorname{tril}_-(L^\mathrm{H} \overline{L}) + \operatorname{triu}(\overline{U} U^\mathrm{H})\\
\overline{A} &= P^\mathrm{T} L^{-\mathrm{H}} \overline{F} U^{-\mathrm{H}}
\end{aligned}$$
}\\

Note that these expressions use the same elementary operations as solving a system of equations using the LU decomposition, as noted above.
The pushforwards and pullbacks can then be computed in-place with no additional allocations.

## Wide $A$

We can write wide $A$ in blocks $A = \begin{bmatrix}A_1 & A_2 \end{bmatrix}$, where $A_1 \in \mathbb{C}^{m \times m}$ and $A_2  \in \mathbb{C}^{m \times (n - m)}$.
It will turn out to be very convenient that $A_1$ is square.
The LU decomposition in terms of these blocks is written
$$P\begin{bmatrix}A_1 & A_2\end{bmatrix} = L \begin{bmatrix}U_1 & U_2\end{bmatrix},$$
where $U_1$ is square upper triangular.
This is a system of two equations that we will address separately.
$$\begin{aligned}
P A_1 &= L U_1\\
P A_2 &= L U_2\\
\end{aligned}$$

### Pushforward

Introducing an intermediate $\dot{H} = \begin{bmatrix} \dot{H}_1 & \dot{H}_2 \end{bmatrix}$ with the same block structure as $U$, the complete pushforward is
$$\begin{aligned}
    \dot{H} &= L^{-1} P \dot{A}\\
    \dot{F} &= \dot{H}_1 U_1^{-1}\\
    \dot{U}_1 &= \operatorname{triu}(\dot{F}) U_1\\
    \dot{U}_2 &= \dot{H}_2 - \operatorname{tril}_-(\dot{F}) U_2\\
    \dot{L} &= L \operatorname{tril}_-(\dot{F})\\
\end{aligned}$$

\details{
Note that the first equation is identical in form to the square LU decomposition.
So we can reuse that solution for the pushforward to get
$$\begin{aligned}
    \dot{F} &= L^{-1} P \dot{A}_1 U_1^{-1}\\
    \dot{L} &= L \operatorname{tril}_-(\dot{F})\\
    \dot{U}_1 &= \operatorname{triu}(\dot{F}) U_1
\end{aligned}$$
Now, let's differentiate the second equation
$$P \dot{A}_2 = \dot{L} U_2 + L \dot{U}_2$$
and solve for $\dot{U}_2$
$$\dot{U}_2 = L^{-1} P \dot{A}_2 - L^{-1} \dot{L} U_2.$$
Plugging in our previous solution for $\dot{L}$, we find
$$\dot{U}_2 = L^{-1} P \dot{A}_2 - \operatorname{tril}_-(\dot{F}) U_2.$$
}\\

### Pullback

Introducing an intermediate $\overline{H} = \begin{bmatrix} \overline{H}_1 & \overline{H}_2 \end{bmatrix}$ with the same block structure as $U$, the corresponding pullback is

$$\begin{aligned}
\overline{H}_1 &= \left(\operatorname{tril}_-(L^\mathrm{H} \overline{L} - \overline{U}_2 U_2^\mathrm{H}) + \operatorname{triu}(\overline{U}_1 U_1^\mathrm{H})\right) U_1^{-\mathrm{H}}\\
\overline{H}_2 &= \overline{U}_2\\
\overline{A} &= P^\mathrm{T} L^{-\mathrm{H}} \overline{H}
\end{aligned}$$

\details{
Here the reverse-mode identity is
$$\Re\ip{\overline{L}}{\dot{L}} + \Re\ip{\overline{U}_1}{\dot{U}_1} + \Re\ip{\overline{U}_2}{\dot{U}_2} = \Re\ip{\overline{A}}{\dot{A}}.$$

We plug in $\dot{L}$, $\dot{U}_1$, and $\dot{U}_2$ to find
$$\begin{aligned}
& \Re\ip{\overline{L}}{L \operatorname{tril}_-(\dot{F})} + \Re\ip{\overline{U}_1}{\operatorname{triu}(\dot{F}) U_1} + \Re\ip{\overline{U}_2}{\dot{H}_2 - \operatorname{tril}_-(\dot{F}) U_2}\\
&= \Re\ip{\operatorname{tril}_-(L^\mathrm{H} \overline{L}) - \operatorname{tril}_-(\overline{U}_2 U_2^\mathrm{H}) + \operatorname{triu}(\overline{U}_1 U_1^\mathrm{H})}{\dot{F}} + \Re\ip{\overline{U}_2}{\dot{H}_2}\\
&= \Re\ip{\left(\operatorname{tril}_-(L^\mathrm{H} \overline{L}) - \operatorname{tril}_-(\overline{U}_2 U_2^\mathrm{H}) + \operatorname{triu}(\overline{U}_1 U_1^\mathrm{H})\right) U_1^{-\mathrm{H}}}{\dot{H}_1} + \Re\ip{\overline{U}_2}{\dot{H}_2}
\end{aligned}$$

Let's introduce the intermediates
$$\begin{aligned}
\overline{H}_1 &= \left(\operatorname{tril}_-(L^\mathrm{H} \overline{L} - \overline{U}_2 U_2^\mathrm{H}) + \operatorname{triu}(\overline{U}_1 U_1^\mathrm{H})\right) U_1^{-\mathrm{H}}\\
\overline{H}_2 &= \overline{U}_2,
\end{aligned}$$
which like $\dot{H}$ we organize into the block matrix $\overline{H} = \begin{bmatrix} \overline{H}_1 & \overline{H}_2 \end{bmatrix}$.
This block structure lets us rewrite the above identity in terms of $\overline{H}$ and $\dot{H}$:
$$\Re\ip{\overline{H}_2}{\dot{H}_2} + \Re\ip{\overline{H}_2}{\dot{H}_2} = \Re\ip{\overline{H}}{\dot{H}}$$

Now we plug in $\dot{H}$
$$\Re\ip{\overline{H}}{\dot{H}} = \Re\ip{\overline{H}}{L^{-1} P \dot{A}} = \Re\ip{P^\mathrm{T} L^{-\mathrm{H}} \overline{H}}{\dot{A}}$$
We have arrived at the desired form and can solve for $\overline{A}$:
$$\overline{A} = P^\mathrm{T} L^{-\mathrm{H}} \overline{H}.$$
}\\

## Tall $A$

The tall case is very similar to the wide case, except now we have the block structure
$$\begin{bmatrix}P_1 \\ P_2 \end{bmatrix} A = \begin{bmatrix}L_1 \\ L_2 \end{bmatrix} U,$$
where now $L_1$ is square unit lower triangular, and $U$ is square upper triangular.
This gives us the system of equations
$$\begin{aligned}
    P_1 A &= L_1 U\\
    P_2 A &= L_2 U.
\end{aligned}$$

### Pushforward

The first equation is again identical to the square case, so we can use it to solve for $\dot{L}_1$ and $\dot{U}$.
Likewise, the same approach we used to solve $\dot{U}_2$ in the wide case can be applied here to solve for $\dot{L}_2$.

Introducing an intermediate $\dot{H} = \begin{bmatrix} \dot{H}_1 \\ \dot{H}_2 \end{bmatrix}$ with the same block structure as $L$, the complete pushforward is
$$\begin{aligned}
    \dot{H} &= P \dot{A} U^{-1}\\
    \dot{F} &= L_1^{-1} \dot{H}_1\\
    \dot{L}_1 &= L_1 \operatorname{tril}_-(\dot{F})\\
    \dot{L}_2 &= \dot{H}_2 - L_2 \operatorname{triu}(\dot{F})\\
    \dot{U} &= \operatorname{triu}(\dot{F}) U
\end{aligned}$$

### Pullback

Introducing an intermediate $\overline{H} = \begin{bmatrix} \overline{H}_1 \\ \overline{H}_2 \end{bmatrix}$ with the same block structure as $L$, the corresponding pullback is
$$\begin{aligned}
\overline{H}_1 &= L_1^{-\mathrm{H}} \left(\operatorname{tril}_-(L_1^\mathrm{H} \overline{L}_1) +  \operatorname{triu}(\overline{U} U^\mathrm{H} - L_2^\mathrm{H} \overline{L}_2)\right)\\
\overline{H}_2 &= \overline{L}_2\\
\overline{A} &= P^\mathrm{T} \overline{H} U^{-\mathrm{H}}
\end{aligned}$$

## Implementation

The product of this derivation is [this pull request to ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl/pull/354), which includes tests of the rules using [FiniteDifferences.jl](https://github.com/JuliaDiff/FiniteDifferences.jl).

## Conclusion

The techniques employed here are general and can be used for differentiation rules for other factorizations of non-square matrices.
A recent paper used a similar approach to derive the pushforwards and pullbacks of the $QR$ and $LQ$ decompositions[^RobertsRoberts2020].

## References

[^HoogAnderssenLukas2011]: de Hoog F.R., Anderssen R.S., and Lukas M.A. (2011) Differentiation of matrix functionals using triangular factorization. Mathematics of Computation, 80 (275). p. 1585. doi: [10.1090/S0025-5718-2011-02451-8](http://doi.org/10.1090/S0025-5718-2011-02451-8).
[^RobertsRoberts2020]: Roberts D.A.O. and Roberts L.R. (2020) QR and LQ Decomposition Matrix Backpropagation Algorithms for Square, Wide, and Deep -- Real or Complex -- Matrices and Their Software Implementation. arXiv: [2009.10071](https://arxiv.org/abs/2009.10071).
