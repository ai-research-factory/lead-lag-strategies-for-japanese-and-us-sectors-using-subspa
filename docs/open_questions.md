# Open Questions

## Phase 1

1. **Covariance estimation period (Cfull)**: The paper uses a fixed period (2010–2014) for covariance estimation. How does this compare to the rolling window approach implemented here? This is the focus of Phase 5.

2. **Open-to-close vs close-to-close returns for Japan**: The paper specifies open-to-close returns for Japanese ETFs. The current validation uses close-to-close returns for simplicity. Phase 2 should implement proper open-to-close return calculation.

3. **Lead-lag alignment**: The current implementation aligns by calendar date intersection, shifting US by 1 day. In production, the alignment should account for actual trading day schedules in both markets (holidays, half-days).

4. **Regularization of covariance matrix**: With 11 features and L=60, the covariance matrix is well-conditioned. However, with shorter windows or more features, regularization (shrinkage) may be needed.

5. **Intercept term**: The current implementation includes an intercept in the regression. The paper's formulation should be checked to confirm whether an intercept is used.
