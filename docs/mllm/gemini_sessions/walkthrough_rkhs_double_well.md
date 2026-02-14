# PoC 10: RKHS-KRONIC Validation (Anisotropic Double Well)

## Objective
Parity check: Can **Pure Kernel EDMD** (scaling with $N$) achieve the same stabilization as the Polynomial/Bilinear approach (scaling with $D$) on the Anisotropic Double Well?

## Methodology
*   **Model**: Kernel EDMD (Gaussian RBF).
*   **Bandwidth**: Automatic Median Heuristic ($\sigma \approx 0.5$).
*   **Samples**: $N=2000$ (Random exploration).
*   **Dimensions**: Infinite feature space (implicitly), reduced to $r=10$ eigenfunctions.
*   **Control**: **Analytic Kernel Gradient** SDRE.
    $$ B_z(x) = \nabla \phi(x) \cdot B_{sys} = \left(\sum v_i \nabla k(x_i, x)\right) B_{sys} $$

## Results
*   **Stabilization**: **Successful**. The controller steers the system from $[-0.5, 0.5]$ to the origin $[0, 0]$ despite the anisotropic noise.
*   **Efficiency**: The online control loop only involves evaluating $N$ kernel functions and solving a tiny $r \times r$ Riccati equation ($10 \times 10$).

![RKHS Parity](/home/ed/.gemini/antigravity/brain/205f126a-3c52-4138-97aa-00356a94422e/rkhs_kronic_double_well.png)

## Comparison with Polynomials
| Feature | Polynomials (PoC 9) | RKHS (PoC 10) |
| :--- | :--- | :--- |
| **Basis** | Monomials (up to deg 3) | Gaussian RBF (Infinite dim) |
| **Complexity** | $O(D^2)$ (Combinatorial) | $O(N)$ (Sample Size) |
| **Control** | Learned Tensor $N_i$ | Analytic Kernel Gradient $\nabla k$ |
| **Flexibility** | Limited to low dim | Universal Approximator |

## Next Steps
Scale up to **CartPole Swing-Up**, where Polynomials typically fail due to global nonlinearities.
