+++
title = "The injectivity radii of the unitary groups"
tags = ["linear algebra", "group theory", "unitary", "rotations", "special orthogonal", "special unitary"]
date = Date(2023, 02, 22)
description = "Working out the injectivity radius for the unitary, orthogonal, special unitary, and special orthogonal groups."
published = date
rss_pubdate = date
rss_description = description
maxtoclevel = 2
+++

_This post was prompted by a user question raised in [this issue](https://github.com/JuliaManifolds/Manifolds.jl/issues/573) on [Manifolds.jl](https://github.com/JuliaManifolds/Manifolds.jl). [Ronny Bergmann](https://ronnybergmann.net/) implemented these results in Manifolds.jl in [this PR](https://github.com/JuliaManifolds/Manifolds.jl/pull/576)._

\tableofcontents

## Introduction

Suppose you held up an object and began rotating it at a fixed speed.
After some amount of time, you stop rotating it.
If I know the starting pose of the object and the amount of time that has passed, under what conditions can I also tell you the rotational velocity (both direction and speed) of the spinning object?

There's really just one condition: that the rotational distance (i.e. angle) between the initial and final positions does not exceed some maximum value.
For example, if you rotate 180° in any direction, it's impossible for me to know whether you rotated the object clockwise or counterclockwise.
Worse, if you rotate a full 360°, I couldn't know whether you didn't move the object at all or performed a whole rotation or a million rotations.
This maximum allowed distance that allows one to still infer the initial and final orientation is called the _injectivity radius_[^1].

The question motivating this post is, what is the injectivity radius for the rotations in not just 2 dimensions and 3 dimensions but any dimension?
And more generally, what is it for the [unitary group](https://en.wikipedia.org/wiki/Unitary_group) and its most common subgroups?
Since these groups are featured in many (all?) of the introductory texts on Lie groups and manifolds, and since the injectivity radius is a basic property introduced in differential geometry textbooks, I was surprised that I could not find a single reference giving these radii for these groups.

In this post I'll work out these radii.
Marvelously, we don't need any differential geometry or group theory to do this, just linear algebra!
Nevertheless, this post assumes familiarity with these topics and for the sake of space will try not define all common terms or notation.[^2]

## The injectivity radius

Consider a point $p$ on some manifold $\mathcal{M}$ with tangent vectors $X,Y \in T_p \mathcal{M}$.
Assume a Riemannian metric $g_p$ defining an inner product $g_p\colon (X, Y) \mapsto \ip{X}{Y}_g$, which induces a norm $\norm{X}_g$.

Let's denote the exponential map $\exp_p\colon X \mapsto q$ for $q \in \mathcal{M}$ and the logarithmic map $\log_p\colon q \mapsto Y$.
The injectivity radius at $p$ is defined as the norm of the smallest $X$ for which $X \ne Y = \log_p (\exp_p X)$, or in other words, the smallest $X$ for which the logarithmic map no longer is the inverse of the exponential map.
We further define the global injectivity radius of $\mathcal{M}$ as the infimum of the injectivity radii at all points on the manifold.

Notationally, we define this global injectivity radius as
$$\operatorname{inj}^-_{\mathcal{M}} = \inf_{(p, X) \in T \mathcal{M}}\{\norm{X}_g | \log_p(\exp_p(X)) \ne X\}.$$

We'll also consider the related supremum
$$\operatorname{inj}^+_{\mathcal{M}} = \sup_{(p, X) \in T \mathcal{M}}\{\norm{X}_g | \log_p(\exp_p(X)) = X\}.$$
These two quantities form the lower and upper bound radii, respectively, of two geodesic balls within which the exponential map is invertible.

## The unitary group(s)

The Unitary group $\mathrm{U}(n, \mathbb{F})$ over some number system $\mathbb{F}$ is the group of all $n \times n$ matrices $p \in \mathbb{F}^{n \times n}$ for which $p^\mathrm{H} p = I_n$, where ${\cdot}^\mathrm{H}$ denotes the matrix adjoint.
$\mathbb{F}$ could be the real numbers $\R$, complex numbers $\C$, or quaternions $\H$.

We will also deal with the following subgroups:
- $\mathrm{SU}(n)$: the special unitary group, which consists of complex unitary matrices whose determinant is +1
- $\mathrm{O}(n) \equiv \mathrm{U}(n, \R)$: the orthogonal group, i.e. the group of rotations and reflections
- $\mathrm{SO}(n)$: the special orthogonal group, i.e. the group of real rotations

We will focus on the real and complex fields, but the unitary quaternionic case immediately follows from the complex one.

## Relevant geometric properties

The unitary group is a compact group and when equipped with the [Frobenius inner product](https://en.wikipedia.org/wiki/Frobenius_inner_product) $g\colon (X, Y) \mapsto \ip{X}{Y}_\mathrm{F}$ becomes a Riemannian manifold.[^3]
The Riemannian exponential $\exp_p$ and logarithm $\log_p$ are related to the Lie group exponential $\operatorname{Exp}$ and logarithm $\operatorname{Log}$, which for these matrix groups are just the matrix exponential and logarithm.

$$
\begin{aligned}
\exp_p(X) &= p\operatorname{Exp}_p(p^{\mathrm{H}}X)\\
\log_p(q) &= p\operatorname{Log}_p(p^{\mathrm{H}}q)
\end{aligned}
$$

@@important
    Thus, to find the injectivity radius at any point $p$, we only need to work out when the matrix exponential is inverted by the principal matrix logarithm.
@@

In the following then, $p$ is always the identity matrix and will not be mentioned, while $X$ is always an element of the Lie algebra, that is, the tangent space at the identity matrix.

The orthogonal group $\mathrm{O}(n)$ is comprised of two submanifolds, $\mathrm{SO}(n)$, whose elements have determinant +1, and another subgroup whose elements have determinant -1.
These submanifolds are disconnected, so that the geodesic cannot join two points from the different submanifolds.
@@important
    The injectivity radius of $\mathrm{O}(n)$ is the same as that of $\mathrm{SO}(n)$.
@@

## Relevant linear algebraic properties

All unitary matrices have a unit determinant $|\det(q)| = 1$.
The inverse of any unitary matrix $q$ is just its adjoint $q^{-1} = q^\mathrm{H}$

The logarithm of any unitary matrix is a skew-hermitian matrix $X = -X^\mathrm{H}$.[^4]
Unitary and skew-hermitian matrices are normal matrices, which means they are always diagonalizable with unitary eigenvectors.
Let $q=VSV^\mathrm{H}$ be the eigendecomposition of $q$ and $X = U \Lambda U^\mathrm{H}$ be the eigendecomposition of $X$.

The unitary/skew-Hermitian condition then implies
$$
\begin{aligned}
X &= -X^\mathrm{H}\\
U \Lambda U^\mathrm{H} &= -U \Lambda^\mathrm{H} U^\mathrm{H}\\
\Lambda &= -\Lambda^\mathrm{H}\\
\diag(\Lambda) = \lambda &= -\lambda^*\\
\lambda + \lambda^* = 2\Re(\lambda) &= 0\\
\lambda_i &= \mathrm{i} \theta_i, \quad \theta_i \in \R, i \in 1\ldots n
\end{aligned}.
$$
@@important
    The eigenvalues of the unitary matrices are points on $\mathrm{U}(1, \mathbb{F})$, and the eigenvalues of the skew-Hermitian matrices are pure imaginary numbers $\lambda = \mathrm{i}\theta$.[^5]
@@

The norm of $X$ under the Frobenius metric is
$$
\begin{aligned}
\norm{X}_g &= \norm{X}_\mathrm{F} = \sqrt{\ip{U \Lambda U^\mathrm{H}}{U \Lambda U^\mathrm{H}}_\mathrm{F}} \\
           &= \norm{\lambda}_\mathrm{F} = \sqrt{\sum_{i=1}^n |\lambda_i|^2} = \sqrt{\sum_{i=1}^n \theta_i^2}\\
           &= \norm{\theta}
\end{aligned}.
$$

The special unitary group has the additional constraint that the determinant is +1.
This constraint implies that $\sum_{i=1}^n \theta_i = 0$.
Given $n-1$ values of $\theta_i$, the $n$th value is thus fixed. 

Special orthogonal matrices are real, and real matrices have the extra property that if they have a complex eigenvalue $\lambda_i$ then they also have a complex eigenvalue $\lambda_i^*$, i.e. [the complex eigenvalues come in conjugate pairs](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Eigenvalues_and_the_characteristic_polynomial).
The sum of the elements in each pair is thus 0, and they don't contribute to the the sum $\sum_{i=1}^n \theta_i$.
When $n$ is odd, at least one eigenvalue is 0.

Using the eigendecomposition, the matrix exponential and logarithm can be computed by applying the corresponding function to the eigenvalues.
Then
$$
\begin{aligned}
\operatorname{Exp}(X) &= U \operatorname{Exp}(\Lambda) U^\mathrm{H}\\
\operatorname{Log}(q) &= V \operatorname{Log}(S) V^\mathrm{H}
\end{aligned}
$$
@@important
    To find the injectivity radii, we only need to find for what values $\theta_i$ is $\exp(\mathrm{i}\theta_i)$ invertible.
@@

## The injectivity radii

We are finally ready to work out the injectivity radii.
For $\mathrm{U}(n, \mathbb{F})$, the eigenvalues of $X$ are $\lambda_i = \mathrm{i} \theta_i$ for $\theta_i \in \R$.

The matrix exponential is invertible when $\exp(\mathrm{i} \theta_i)$ is invertible.
But this is just rotation in the complex plane by the angle $\theta_i$, which is invertible for $\theta_i \in (-\pi, \pi]$.
The injectivity radii are then computed by the constraint $|\theta_i| = \pi$ for at least one $i \in 1\ldots n$. 

For $\mathrm{U}(n, \mathbb{F})$ and $\mathbb{F} \not\equiv \R$, the largest value of $\norm{\theta}$ achievable subject to the constraints occurs when $|\theta_i| = \pi$ for all $i$.
Then $\norm{\theta} = \pi \sqrt{n}$.
The smallest value of $\norm{\theta}$ occurs when $|\theta_i| = \begin{cases}\pi, & i = k\\ 0, & i \ne k\end{cases}$ for any $k \in 1\ldots n$.
So
@@important
$$
\begin{aligned}
\operatorname{inj}^-_{\mathrm{U}(n, \mathbb{F})} &= \pi, \qquad \mathbb{F} \not\equiv \R\\
\operatorname{inj}^+_{\mathrm{U}(n, \mathbb{F})} &= \pi \sqrt{n}.
\end{aligned}
$$
@@

For $\mathrm{SU}(n)$, we have the additional constraint that $\sum_{i=1}^n \theta_i = 0$.
For even $n$, $\norm{\theta}$ is maximized subject to these constraints when $n/2$ entries in $\theta$ are $+\pi$ and when $n/2$ entries are $-\pi$.
On the other hand, for odd $n$, it is maximized when $(n-1)/2$ values each are $+\pi$ and $-\pi$ and the $n$th value is $0$. 
$\norm{\theta}$ is minimized subject to these constraints when there is a single nonzero pair $\theta_i = -\theta_j = \pi$ for $j \ne i$.
As a result,
@@important
$$
\begin{aligned}
\operatorname{inj}^-_{\mathrm{SU}(n)} &= \pi\sqrt{2}\\
\operatorname{inj}^+_{\mathrm{SU}(n)} &= \pi \sqrt{2 \lfloor n/2 \rfloor}.
\end{aligned}
$$
@@

The constraints on $\theta$ required for $\mathrm{O}(n)$ and $\mathrm{SO}(n)$ are also satisfied by the lower and upper bounds of the norms considered for $\mathrm{SU}(n)$, so
@@important
$$
\begin{aligned}
\operatorname{inj}^-_{\mathrm{O}(n)} = \operatorname{inj}^-_{\mathrm{SO}(n)} &= \pi\sqrt{2}\\
\operatorname{inj}^+_{\mathrm{O}(n)} = \operatorname{inj}^+_{\mathrm{SO}(n)} &= \pi \sqrt{2 \lfloor n/2 \rfloor}.
\end{aligned}
$$
@@

It's always a good idea to numerically check that the results make sense.
For the 2D and 3D rotations $\mathrm{SO}(2)$ and $\mathrm{SO}(3)$, respectively, we then have
$$
\begin{aligned}
\operatorname{inj}^-_{\mathrm{SO}(2)} = \operatorname{inj}^+_{\mathrm{SO}(2)} = \pi\sqrt{2}\\
\operatorname{inj}^-_{\mathrm{SO}(3)} = \operatorname{inj}^+_{\mathrm{SO}(3)} = \pi\sqrt{2}
\end{aligned}
$$
These injectivity radii correspond to a rotation of 180°. [^3]
Coming back to our motivating example, if we start from any pose and rotate an object more than 180° in any direction, we can no longer uniquely determine the initial rotational velocity.

$\mathrm{U}(1, \C)$, the complex unit circle, is isomorphic to ${\mathrm{SO}(2)}$, so it may seem like a contradiction that its injectivity radius is $\pi$.
But this difference is again caused by the choice of metric, which causes the inner product on ${\mathrm{SO}(2)}$ to be scaled by $\frac{1}{2}$ compared to $\mathrm{U}(1, \C)$.[^3]

Analogously, $\mathrm{U}(1, \H)$ represents the unit quaternions (also the compact symplectic group), which are an alternative way to represent 3D rotations, and is equivalent to $\mathrm{SU}(2)$.
The injectivity radii again only differ by the factor of $\sqrt{2}$ due to the scaling convention of the metric.

## Conclusion

This has been one of the rare moments where we got to dabble with group theory and manifolds without needing too much geometry.
I hope it was enjoyable!

###

[^1]: The injectivity radius is so called because it is the radius of a geodesic ball within which the exponential map is injective (one-to-one).
[^2]: It's _really_ hard to write anything about manifolds or groups without writing a whole introductory text to differential geometry or group theory.
[^3]: Because of the skew-hermitian nature of the elements of the Lie algebra, some texts use the scaled Frobenius metric $g(X, Y) = \frac{1}{2}\ip{X}{Y}_\mathrm{F}$. In some cases, this allows the norm of a tangent vector to be interpreted as the angle of the rotation. To get the injectivity radii for this metric, one would just divide ours by $\sqrt{2}$.
[^4]: This comes from differentiating the constraint $q^\mathrm{H}q=I_n$. Then we have $\dd{(q^\mathrm{H}q)} = (\dd{q})^\mathrm{H}q + q^\mathrm{H} \dd{q} = \dd(I_n) = 0$. Letting $X = \dd{q}$, then $X^\mathrm{H}q = -q^\mathrm{H} X = -(X^\mathrm{H}q)^\mathrm{H}$. When $q$ is the identity matrix, $X = -X^\mathrm{H}$.
[^5]: For the complex unitary group, $\mathrm{i}$ is the usual imaginary number, while for quaternions, it would be a pure unit quaternion.
