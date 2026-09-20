# FROZEN LINEAGE

The empirical record of the manuscript lives in twelve frozen
namespaces. The table below is transcribed from the frozen record
(`PAPER2_MASTER_HANDOFF.md`, section 8.1); the local workspace base
path of that record is omitted. Across Phase 0B-2E the frozen
re-check reported 0 mismatches against each
`artifact_sha256sums.txt`.

| namespace dir | files | sums entries | sha256(artifact_sha256sums.txt) | manifest sha256 / entries |
|---|---|---|---|---|
| late_reversal_early_eval_phase0a | 56 | 55 | 3b66c02b0ccba723805bb7b7f39da74c35d1aad58c74c7dc897d0d9a301df5f1 | (no manifest.json; integrity aaf33490fde738d963158860736abf2b4639ccac71a3f561c61506572ce24410) |
| late_reversal_early_eval_phase0b_signal_hunt | 60 | 59 | adf13e644a90b0e33e2de51960df1bdf932e53d9264e6022acfd3d917a06b465 | e1ca87e21914a35a27bd530f1cae2d02d52f544d992990857b624c261f8d02c2 / 57 |
| late_reversal_early_eval_phase0c | 22 | 21 | 2144daf29396e396f2c10cecc4fcf095627aed4ade4920ba0898acc25abeb268 | 94d0a649ef5f64772b786432e9472f4b3b42b6cf4a18239f88afeb2823a1e481 |
| earlyeval_phase0d_cross_agent_calibration | 19 | 18 | 790e09f762477d538a53df8f83fb419955889c513269f4fbbd652c322c3f5c78 | 90e79c9177b03f74b5989e59c0d267f67cb53d3ba9a6b8e3fd9adab8b13998cd |
| earlyeval_phase0e_prior_shift_decomposition | 16 | 15 | 40331b1c1202a75907576e1d67b7fcd9953a637fe8963ab11d320794b6ee3959 | a58b2d48ae73235b0ed68190f6d054a0505ac7cdbab4c855376e06904b3ebed7 |
| earlyeval_phase1a_same_predictor_transfer | 200 | 199 | d231809da36900639418bc1387d31b0610765971b38bce72766cd83e359091ce | 29d3fe47aa9049f1c11db12fe9e11d1eea5f59fa03ee515d035fe978647325ae / 198 |
| earlyeval_phase1a_d_target_persistence | 15 | 14 | 0b49562822a5f6a3fbc3b311b1827ee4995478123fb4357f46ebeb9aa2530ab5 | b72c3653756e31c61d5794e22e47ad8f35ac8812946c6224ea559d9c5d7343c5 / 13 |
| earlyeval_phase1b_threshold_robustness | 12 | 11 | 483d60525abf3e7af848440b97863e093634b588c594962dd71b160ec5f52213 | e50162b7d31da79701cb2848a7e47aac5d762f63abf8aee2e7e39004442ac878 / 10 |
| earlyeval_phase2a_terminalbench_coverage | 21 | 19 | f15d964e47e4996b9c1caf8f75e66ebe347cfebdc1e6b86c63b506170d40bc2e | 0ceb8f48d353fa230b40d3750728ee0b25161d9896af6d16ad8e2324397db41a |
| earlyeval_phase2b_terminalbench_fixed_scaffold | 108 | 107 | d2edab18d4ab44b158837e7a4df41dbdd7e173dfd6f6e277199eecab64a99263 | 14e02040f39eb496af5884b6a3c06421d276e7634ec7c8b6899594392456ebb4 / 106 |
| earlyeval_phase2b_d_threshold_diagnostic | 12 | 11 | 82491a5193e77e9df0fa81602870f200f844eaa20a7437ead86bfd30eb5e3391 | a67a4cdb0140ec363eb3a1153c04348617c21640cd78a32c03decd95763b70ac / 10 |
| earlyeval_phase2e_exact_target_cross_benchmark | 10 | 9 | d35f3ce01eddcb72bc2a1a3a0c43ab8522cee9413f039015d45595ba32d97bf2 | d6344b7aad2bb8de5077427d6aa6365e5ba82036f6fb2bdbf792af5604799d40 / 8 |

Integrity-report sha256 per phase (cross-reference):

    0a    aaf33490fde738d963158860736abf2b4639ccac71a3f561c61506572ce24410
    0b    40e7acfb0577ef1a7575dfa61687497bf57e52c8809b4ca5a43694b76e83c5eb
    0c    c4ed57a9fafeac0d68ae98e4e175f831b3cdd43b8d84dc7318854eed11bbdf25
    0d    7c0f03e2957cd451bc60758d934eeb41dce2ddc3b4e2f5c8b5f53f250dc8b4f4
    0e    46f34c77f032dd0fab42a618c9ec70c940165b76c55578a5009a388f511e8ddd
    1a    394777715faf0a9555e4cb674413f89c56dfa1af30e877ee69cdf6e093b2a7f1
    1a-d  d1acac637aea1ef43b6642e6a3d914797cbe8cadb6b84c268cdd6ec1fc64b001
    1b    81493b2c97de4f9556d598799d78f3a370795f197760b916c011881702317163
    2a    a8009b93f261a48979599c253aa8fced37c3952914c98927107ac119fddffb0d
    2b    63b846f57a91cfcbbd8ee97bc69cc0c05a673db9662134ab9f62c4e9ae96d031
    2b-d  f102b2f76c90b0a7d48d405216805fa3b9f9b147e7932fea2397812ce365dbc5
    2e    bebc9bb4eb2d338446a751605d86cccbb1e56673d861ce434c690c444714068f

Per-namespace report files (source of the recorded numbers):

    late_reversal_early_eval_phase0a/FEASIBILITY_REPORT.md
    late_reversal_early_eval_phase0b_signal_hunt/SIGNAL_REPORT.md
    late_reversal_early_eval_phase0c/PHASE0C_REPORT.md
    earlyeval_phase0d_cross_agent_calibration/PHASE0D_REPORT.md
    earlyeval_phase0e_prior_shift_decomposition/PHASE0E_REPORT.md
    earlyeval_phase1a_same_predictor_transfer/PHASE1A_REPORT.md
    earlyeval_phase1a_d_target_persistence/PHASE1A_D_REPORT.md
    earlyeval_phase1b_threshold_robustness/PHASE1B_REPORT.md
    earlyeval_phase2a_terminalbench_coverage/PHASE2A_REPORT.md
    earlyeval_phase2b_terminalbench_fixed_scaffold/PHASE2B_REPORT.md
    earlyeval_phase2b_d_threshold_diagnostic/PHASE2B_D_REPORT.md
    earlyeval_phase2e_exact_target_cross_benchmark/PHASE2E_REPORT.md

Each namespace also carries `PROTOCOL.md`, `manifest.json`,
`integrity_report.json` and `artifact_sha256sums.txt`; the Phase 0A
namespace carries `FEASIBILITY_PROTOCOL.md` and no `manifest.json`.

The protocols of these namespaces are shipped in `protocols/`; the
aggregate derived files are shipped in `derived_data/` with their
per-file source mapping in `derived_data/PROVENANCE.md`. The full
row-level prediction tables and the third-party corpora are not
redistributed.
