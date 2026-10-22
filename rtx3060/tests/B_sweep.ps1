Set-Location C:\Users\sebinj\qwen_inference
$c=@("--staged","12","--snapshot","kv-200k-code-u2048-q4-mtp.bin","--n","512","--reps","1","--env","LLAMA_FA_VEC_MAX_NQ=128","--env","LLAMA_FA_DECODE_Q4=1","--env","LLAMA_RS_REPLAY=1","--env","LLAMA_MTP_WINDOW=8192","--env","LLAMA_MMVQ_BATCH_INVARIANT=1","--draft-ubatch","16")
# prefill A/B (no MTP): vec prefill (dq4 only for n_q<=8) vs dq4 prefill
python tests\dq4_e2e.py @c --suffix-file results\dq4\task3.suffix.txt --env LLAMA_FA_DECODE_Q4_MAXQ=8 --tag "B2-t3-vecpf-nomtp" --out results\dq4\B2-t3-vecpf-nomtp.json
python tests\dq4_e2e.py @c --suffix-file results\dq4\task3.suffix.txt --tag "B2-t3-dq4pf-nomtp" --out results\dq4\B2-t3-dq4pf-nomtp.json
# ~1K-token turn, full MTP config, and its no-MTP reference
python tests\dq4_e2e.py @c --suffix-file results\dq4\task3long.suffix.txt --spec draft-mtp --draft-max 4 --tag "B2-t3long-d4" --out results\dq4\B2-t3long-d4.json
python tests\dq4_e2e.py @c --suffix-file results\dq4\task3long.suffix.txt --tag "B2-t3long-nomtp" --out results\dq4\B2-t3long-nomtp.json
# larger ubatch
foreach ($ub in @(256,512)) {
  python tests\dq4_e2e.py @c --suffix-file results\dq4\task3long.suffix.txt --spec draft-mtp --draft-max 4 --server-arg=-ub --server-arg=$ub --server-arg=-b --server-arg=$ub --n 64 --tag "B2-t3long-d4-ub$ub" --out "results\dq4\B2-t3long-d4-ub$ub.json"
}
