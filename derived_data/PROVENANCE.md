> **Release note (PAPER2_W7B).** In this repository these derived files
> live under `derived_data/` (the pre-finalization package used the
> directory name `repository_candidate/`). File contents are unchanged
> except for the single corrected literal documented in
> `derived_data/PROVENANCE.md`, and the references to the author-decision
> files of the pre-finalization workspace, which this release replaces
> with the release level `README.md` / `LICENSE` / `LICENSE-DATA`.

Every file below was copied byte-for-byte from the frozen workspace namespace
named on the right. SHA-256 values are of the copies in this package.

## Source mapping

    data/figure2_plot_data.csv  <-  outputs/paper2_w2_visual_architecture/data/figure2_plot_data.csv
    data/figure3_plot_data.csv  <-  outputs/paper2_w2_visual_architecture/data/figure3_plot_data.csv
    data/figure3_annotation_values.json  <-  outputs/paper2_w2_visual_architecture/data/figure3_annotation_values.json
    data/figure4_threshold_data.csv  <-  outputs/paper2_w2_visual_architecture/data/figure4_threshold_data.csv
    data/figure4_cross_benchmark_data.csv  <-  outputs/paper2_w2_visual_architecture/data/figure4_cross_benchmark_data.csv
    data/table1_design_data.csv  <-  outputs/paper2_w2_visual_architecture/data/table1_design_data.csv
        (release note: one literal corrected - the TerminalBench dataset revision
        `04e8940f5b6736a7ce2d22224fe2f2af74163ed2` printed in the frozen W2 design
        data was a transposition of the revision actually consumed,
        `04e8940f5b6736a7ce8d22224fe2f2af74163ed2`; the frozen copy is unchanged and
        this release copy carries the corrected revision)
    data/table2_primary_targets.csv  <-  outputs/paper2_w2_visual_architecture/data/table2_primary_targets.csv
    data/table3_terminal_boundary.csv  <-  outputs/paper2_w2_visual_architecture/data/table3_terminal_boundary.csv
    data/phase0d_calibration_gaps.csv  <-  outputs/earlyeval_phase0d_cross_agent_calibration/analysis/calibration_gaps.csv
    data/phase0d_wilson_intervals.csv  <-  outputs/earlyeval_phase0d_cross_agent_calibration/analysis/wilson_intervals.csv
    data/phase0d_per_model_reliability.csv  <-  outputs/earlyeval_phase0d_cross_agent_calibration/analysis/per_model_reliability.csv
    data/phase0e_corrected_gaps.csv  <-  outputs/earlyeval_phase0e_prior_shift_decomposition/analysis/corrected_gaps.csv
    data/phase0e_prior_correction.csv  <-  outputs/earlyeval_phase0e_prior_shift_decomposition/analysis/prior_correction.csv
    data/phase0e_residual_gap_range.json  <-  outputs/earlyeval_phase0e_prior_shift_decomposition/analysis/residual_gap_range.json
    data/phase1a_pairwise_metrics.csv  <-  outputs/earlyeval_phase1a_same_predictor_transfer/analysis/pairwise_metrics.csv
    data/phase1a_target_persistence.csv  <-  outputs/earlyeval_phase1a_same_predictor_transfer/analysis/target_persistence.csv
    data/phase1a_primary_gate.json  <-  outputs/earlyeval_phase1a_same_predictor_transfer/analysis/primary_gate.json
    data/phase1ad_partner_jackknife.csv  <-  outputs/earlyeval_phase1a_d_target_persistence/analysis/partner_jackknife.csv
    data/phase1ad_primary_gate.json  <-  outputs/earlyeval_phase1a_d_target_persistence/analysis/primary_gate.json
    data/phase1b_target_threshold_summary.csv  <-  outputs/earlyeval_phase1b_threshold_robustness/analysis/target_threshold_summary.csv
    data/phase1b_secondary_descriptives.csv  <-  outputs/earlyeval_phase1b_threshold_robustness/analysis/secondary_descriptives.csv
    data/phase2a_trajectory_usability.csv  <-  outputs/earlyeval_phase2a_terminalbench_coverage/analysis/trajectory_usability.csv
    data/phase2b_eligible_models.json  <-  outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/analysis/eligible_models.json
    data/phase2b_per_target_metrics.csv  <-  outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/analysis/per_target_metrics.csv
    data/phase2b_target_signal.json  <-  outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/analysis/target_signal.json
    data/phase2b_robust_targets.csv  <-  outputs/earlyeval_phase2b_terminalbench_fixed_scaffold/analysis/robust_targets.csv
    data/phase2bd_denominator_summary.csv  <-  outputs/earlyeval_phase2b_d_threshold_diagnostic/analysis/denominator_summary.csv
    data/phase2bd_robust_targets.csv  <-  outputs/earlyeval_phase2b_d_threshold_diagnostic/analysis/robust_targets.csv
    data/phase2bd_diagnostic_gate.json  <-  outputs/earlyeval_phase2b_d_threshold_diagnostic/analysis/diagnostic_gate.json
    data/phase2e_target_cross_benchmark.csv  <-  outputs/earlyeval_phase2e_exact_target_cross_benchmark/analysis/target_cross_benchmark.csv
    data/phase2e_threshold_descriptives.csv  <-  outputs/earlyeval_phase2e_exact_target_cross_benchmark/analysis/threshold_descriptives.csv
    metadata/claim_evidence_matrix.csv  <-  outputs/paper2_w1_claim_framing/claim_evidence_matrix.csv
    metadata/figure_source_map.csv  <-  outputs/paper2_w2_visual_architecture/figure_source_map.csv
    metadata/table_source_map.csv  <-  outputs/paper2_w2_visual_architecture/table_source_map.csv

## SHA-256 of the copies

    data/figure2_plot_data.csv  926916531bd3ac99019a99459e5a3ac1dbce3d9f31cef9ee7481eb614bab86cb  (2953 bytes)
    data/figure3_plot_data.csv  a161c600125cecbec2cb5def05bf039e5dcca03c085873aaac6a4381e2d16a59  (3617 bytes)
    data/figure3_annotation_values.json  1bf75b9dac44fab4e3fea659d7bcb3dd68eeee38213fb4e34e8ce532f8da1ebd  (3212 bytes)
    data/figure4_threshold_data.csv  2908f56bea74bcdb1c405620e063690c50c33b5cf6937098c7f7e9b3e8021962  (1671 bytes)
    data/figure4_cross_benchmark_data.csv  1c82d22b0ef164fcc2b01c0ddc401e54a75fe61740182cffbf67a568a9ee6841  (1174 bytes)
    data/table1_design_data.csv  99feea15afcd9e464cacbd3ad9ceb09647b2efd3c1315c80117f792a300db3d6  (1267 bytes; corrected revision literal as noted above; the frozen
      W2/W7A copy of this file remains 0da47a1a17f307f1c94f9534674ff82c7998016bc16e23bec28271c6bb9267e7)
    data/table2_primary_targets.csv  a81bd39fa5d59bba3b94aa32caa08bc44341de9d7c32fc7bf89f934fa251b565  (829 bytes)
    data/table3_terminal_boundary.csv  d86cc948be7256a7a858daa2ec5c5e9169056ac154efd72d442d2c3960e12f41  (1443 bytes)
    data/phase0d_calibration_gaps.csv  d18a87419380667c642c4771a3d275b3d39c5c5e3207d41b267b024e3670d250  (2541 bytes)
    data/phase0d_wilson_intervals.csv  ccc35bf8f2a6126cf132e252c0df44a8acc6b54d54c8e5788ec18e7ea80439d3  (4343 bytes)
    data/phase0d_per_model_reliability.csv  ff3cced44bd204c5558315110073611777748b6c7333537c46bbd51bfe49fa88  (1329 bytes)
    data/phase0e_corrected_gaps.csv  e509cba1e2ec0a2446185693e1f2b873a69b1ce95b212beff31fd024171dd47e  (3827 bytes)
    data/phase0e_prior_correction.csv  d2dd0f7f8b87752c8b596f4669b92755757fa01588400358208b555a463a6fca  (1939 bytes)
    data/phase0e_residual_gap_range.json  db732e9e314d9e7c53249f5896320590ff75a1224cdab22b8a13b0b3225db737  (2711 bytes)
    data/phase1a_pairwise_metrics.csv  1d5850ffe854e5fb014a79bcd6936cf5c8c9790bbc98400044fa66accd068024  (56351 bytes)
    data/phase1a_target_persistence.csv  c2dc972bac53e487b3c62bd638c81e5676f4cec9723b3d5f833005579f29de03  (2799 bytes)
    data/phase1a_primary_gate.json  cfdf2a889fd3c0e0f89e18d7ce5a1fa4555859a5722225685c9252e0732a187a  (74111 bytes)
    data/phase1ad_partner_jackknife.csv  37ea21f03d39363ee48394084638b07467c7e05c0d545f24869dd8bdcccab187  (2622 bytes)
    data/phase1ad_primary_gate.json  24d5a3c8531d74e1f3d19bafb0d7b1700fa45572c0069add6ffab8a940c5cb8b  (4576 bytes)
    data/phase1b_target_threshold_summary.csv  e966d756892c18917e04240408882463ba345f26c2b86666d13fdfc15261edca  (2178 bytes)
    data/phase1b_secondary_descriptives.csv  8d5a81f5a3c76d4301456565c355becf93a3d24183e77ab157476b197f33b5a9  (1488 bytes)
    data/phase2a_trajectory_usability.csv  22c3aef2a80c2762847bc0e759435b44a36b5e12e081da991c7f851816a197ff  (7345 bytes)
    data/phase2b_eligible_models.json  aeb70670bb807a9af72fd1e3e8d7985ecc278cd7d917808ba8fa6d553fa40cbd  (6898 bytes)
    data/phase2b_per_target_metrics.csv  eaed3b326c7781270a1fc37d2be34e71adc6488a59289ef561194058efccf7ab  (18298 bytes)
    data/phase2b_target_signal.json  b510a0a2c7a748d27fe5a3715c04f6ca533330242c1a524e3324d307346a16f3  (3015 bytes)
    data/phase2b_robust_targets.csv  904016153485a5e490e98809c7a9c3e12ec8db509be265b7c26d964dcb3fd928  (503 bytes)
    data/phase2bd_denominator_summary.csv  9d2f6020baf1770053a8ab5d2cb386261b6be80ccaf01ac5705e5cb41bcedca9  (396 bytes)
    data/phase2bd_robust_targets.csv  76a6d231e3dd20dceca0ee33ce719a4f031ce033be843b36aad67db8b73429f1  (1967 bytes)
    data/phase2bd_diagnostic_gate.json  4cc2caa95622e2241b5d5a2067ba7bd503999fee0c7749396a2d6b54bce22644  (5290 bytes)
    data/phase2e_target_cross_benchmark.csv  c5f430e8f6d87ff88d854f22133641b55e2f8091c48dec2647b450885c22aba8  (1510 bytes)
    data/phase2e_threshold_descriptives.csv  5833f376a59ab233dc8378026019f673d0ddbca3d8b89c2fc8d3416ec67a533d  (2142 bytes)
    metadata/claim_evidence_matrix.csv  83c415dab363a4beb3faedd13012add4ca1b580138010fb3757a1f30fc23050b  (15478 bytes)
    metadata/figure_source_map.csv  27fba348c2ffa7c9c9a1bccd071ab317d0970d6738e890840b6a406f78efcce4  (26849 bytes)
    metadata/table_source_map.csv  aff47132f732ba0b6afa1008cdb5e0d758cbafe50fa5c867ee6827267712d1b1  (14381 bytes)

## Integrity of the frozen originals

The originals live in `outputs/` namespaces that each carry `manifest.json`,
`integrity_report.json` and `artifact_sha256sums.txt`. At the close of the
empirical record all namespaces re-verified with 0 mismatches against their own
checksum files; the per-namespace checksum-file hashes are recorded in
`PAPER2_MASTER_HANDOFF.md` section 8.1 of the frozen record.
