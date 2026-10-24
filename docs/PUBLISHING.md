# Repository and release

Repository: https://github.com/Mubaoy/mHCMTN_CODE

Initial version: `v0.1.0`; current version: `v0.1.1` (datetime-resolution fix).
The repository root contains the contents of the local
`mHCMTN_CODE` directory, not its surrounding workspace. Raw workbooks, weights,
experiment caches and local credentials are excluded. The source is independent
of its parent directory.

The author has not selected an open-source reuse license; see
[LICENSE_NOTICE.md](../LICENSE_NOTICE.md). Public visibility does not itself
grant an open-source license or rights to third-party datasets.

## Updating the code

Run tests and inspect the staged changes before pushing:

```powershell
python -m pytest -q
git status --short
git add <files-to-publish>
git diff --cached
git commit -m "Describe the change"
git push origin main
```

Keep the initial version tag fixed. For later scientific results, publish a new
version/tag and cite the exact code commit that generated those results. Record
the actual configuration, split, seeds and environment with each experiment.

The availability statement is in `CODE_AVAILABILITY.txt` and
`docs/AVAILABILITY_TEMPLATE.md`. Historical figure reconstruction is not a claim
that the new code reproduces historical results; the method and validation notes
define the scope of this release.
