# Synthetic stored-build fixture

`stored-build-http-v1.json` is a byte-identical copy of Server
`a421c48632835b14242ad58f3827462f4af5b374`:
`tests/fixtures/prompt-build-http-v1.json`.

File SHA-256: `5efc2497eac64c41fcdca329f085875d396be6d9ffe9f4bfff48e0e65aa83cb7`.
The fixed compiler produced its canonical build body and content identity. Native/account/audit
IDs are deliberately fictional. It proves no authorization, provider use or installed source
attestation. No operational/research prompt, secret, model asset or actual account is included.

The existing shared byte/render fixtures remain in `tests/fixtures/prompts/` at the repository
root and keep their original bytes and identities. Package tests read those authoritative files;
the runtime package never reads fixtures or files. Tests require a source checkout, not an
unbuilt distribution or a separately downloaded dataset.
