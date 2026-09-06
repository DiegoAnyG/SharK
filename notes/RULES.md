# Rules

Hard constraints. A rule here outranks anything convenient, any deadline, and any earlier
instruction that conflicts with it.

## 1. Never bend the code toward a result

Never write code so that a chosen answer comes out. Not a tautomer ranked as the most stable, not an
energy difference reaching a desired target, not a geometry matching a previous run or an expectation.

**Allowed, and encouraged:** correcting a fault, optimising, adding a method, changing a parameter
for a stated methodological reason. Making conformer generation deterministic with a fixed seed is
allowed — it fixes *which* structure is sampled, not *what score or energy it earns*. Seeding RDKit
ETKDG or initial coordinate generation is allowed for the same reason: it removes stochastic noise,
it does not choose an outcome.

The line: **a change may decide what the method is, never what the answer is.** If a change would
be embarrassing to describe in the Methods section of a publication, it is the wrong change.

A test that asserts a specific absolute energy is a trap — it invites making the code fit the number.
Assert *reproducibility* (two runs agree), *convergence*, and *physical properties*, not arbitrarily
chosen values.

## 2. Never leak anything personal

No absolute paths from a personal machine, no usernames, no machine names, no home directories,
no credentials, in code, comments, tests, docs, commit messages, or these notes.

**The one exception:** the author's initials and surname, for citation and authorship.

Discover paths at runtime. Where a path must be written down, use `~/shark`,
`<projects root>`, `<install prefix>`, `<scratch dir>`.

## 3. Never hardcode

No hardcoded paths, no hardcoded lists of files, molecules, functional/basis sets, or tools. Discover from
disk or configuration. Honour environment variables. The software must work on a machine nobody anticipated.

Environment variables to honour:
- `SHARK_ORCA` (path to ORCA executable or directory)
- `SHARK_SCRATCH` (path to scratch / temporary calculation directory)
- `SHARK_NPROCS` (default processor cores to allocate)
- `SHARK_MAXCORE` (memory allocation in MB per core)

A path that only exists on the machine the code was written on is the specific failure this rule
exists to prevent: it does not error, it silently reports that everything is absent.

## 4. English, always

All code, comments, docstrings, identifiers, on-disk names, tests, documentation, commit messages
and these notes are written in English.

**Deliberately Spanish, do not translate:** user-facing documentation/catalogues where localization
is specifically implemented, author name, or explicit external legacy naming.

## 5. Keep the workspace clean

- Announce every new directory: where it is, what it is for, and whether it can be deleted.
- Do not leave scratch files (`.tmp`, temporary ORCA files like `.gbw`, `.densities`, `.cpcm` when unneeded) in the project tree.
- Temporary / high-I/O scratch work goes in a designated scratch space outside the repository.

## 6. Backward compatibility is mandatory

Older calculation projects, serialized configs, and schema outputs must keep opening and parsing.
Add to legacy adapters and layout resolution rather than renaming or breaking fields in place.

## 7. Verify what only shows up at runtime

Changes that cannot fail at import time — quantum chemistry binary execution, process termination
and signal handling, memory allocation limits, parser handling of varied ORCA output versions,
and Windows/Linux cross-platform paths — get a test.

## 8. Commits as changelogs

Every git commit message must function as a concise changelog. Use a clear summary title line, followed by bulleted items listing only the most important changes (between 1 and maximum 2 lines per item). Keep descriptions brief and informative.

