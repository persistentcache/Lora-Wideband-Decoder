# Radio workbench implementation plan

Goal: replace the crowded, brightly colored UI with a restrained dark instrument interface. User delegated design decisions.
Architecture: retain Flask and the existing browser renderers/API contracts. Reorganize the page shell and filters, extract visual rules to a static stylesheet, and use consistent neutral surfaces across all views.

- [x] Replace the header/navigation with a compact sidebar, receiver controls, session counters, and descriptive view headings.
- [x] Consolidate packet filters into a native select without changing existing filter semantics. Keep search and merge controls visible.
- [x] Apply charcoal surfaces, muted status colors, tabular numeric typography, consistent forms and responsive layouts throughout settings, sessions and data views.
- [x] Verify browser rendering at desktop/mobile widths, packet filtering/expansion, navigation, settings and sessions using synthetic data; run existing non-hardware tests and inspect the diff.

Review focus: empty versus filtered states; long packet contents; narrow-screen horizontal overflow; keyboard navigation; unchanged API actions. No decoder changes or new runtime dependencies. Native worktree creation failed because task cwd is home; isolated Git worktree created beside existing checkout.

Validation: `python3 -m pytest -q` — 30 passed. Chromium checks against Flask at 1440×960, 390×844, and 844×390 passed: all eight views, packet content filter, text search, row expansion, configuration, sessions, skip-link view preservation, sidebar scrolling and overflow. No browser JavaScript errors. Synthetic packet data used only in the browser; live SDR reception was not exercised. `git diff --check` passed. Independent review identified skip-link hash handling and short-sidebar scrolling; both fixed and rechecked.
