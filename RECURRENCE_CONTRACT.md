# Canonical recurrence contract

1. Every recurrent tensor is `[batch, sequence_length, features]` and
   `sequence_length > 1`.
2. Time is strictly increasing within each complete trajectory or window.
3. Only complete trajectories or complete windows may be shuffled.
4. Every derived window retains its persistent source trajectory identity.
5. Train, validation, and test source identities are pairwise disjoint.
6. Recurrent state starts at `None` for every independent trajectory batch.
7. State crosses only contiguous truncated backpropagation through time (TBPTT) chunk boundaries belonging to that same batch and
   is detached at each boundary. A would-be one-sample remainder is absorbed into the
   preceding chunk, so no sample is dropped and no recurrent chunk has length one.
8. Validation determines architecture and cluster count. Test data is revealed only after
   selection and is never used for hyperparameter choice.
9. Observation-weighted MSE and BIC both count scalar target elements.
10. Batch export rejects length-one sequences; streaming export advances explicit state
    one physical sample at a time.
11. Compiled circuit export uses a two-phase sample clock and capacitive hidden state.
    One simulator process represents one independent trajectory; feature inputs are
    zero-order held and state advances exactly once per chronological sample.

The test suite treats every item above as a regression barrier. Static MLP, PIKAN, and
SINDy implementations are explicitly labeled baselines; they accept chronological
sequence-shaped data but intentionally have no recurrent state.
