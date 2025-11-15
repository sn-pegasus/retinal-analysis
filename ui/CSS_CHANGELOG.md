# CSS Change Log

This document tracks visual updates and cleanup applied to the redesigned UI.

Date: 2025-10-25
Author: UI Maintenance

## Viewports Container Cleanup
- File: `ui/styles.css`
- Section: Viewports container definitions
- Changes:
  - Removed `!important` from `.viewports-container` grid properties:
    - `display: grid` (was `display: grid !important`)
    - `grid-template-columns: 1fr` (was `1fr !important`)
    - `grid-template-rows: 1fr 1fr` (was `1fr 1fr !important`)
  - Removed `!important` from `.viewports-container.side-by-side`:
    - `grid-template-columns: 1fr 1fr` (was `1fr 1fr !important`)
    - `grid-template-rows: 1fr` (was `1fr !important`)
- Rationale: Allow layout toggling and focus mode to work via natural specificity without forced overrides.

## Focus Mode Simplification
- File: `ui/styles.css`
- Section: `.viewports-container.focus-mode` and child `.viewport`
- Changes:
  - Removed `!important` across width and flex properties.
  - Ensured focus mode relies on selector specificity (no legacy overrides).
- Rationale: Maintainable styling that avoids cascade conflicts and improves cross-browser consistency.

## Behavioral Reinforcement
- File: `ui/app.js`
- Section: `toggleViewportLayoutSimple`
- Changes:
  - After layout toggle, call `showE2EControls()` when `isE2EMode` to ensure eye tree containers remain visible across layout changes.
- Rationale: Preserve E2E tree visibility and UI positioning when switching stacked/side-by-side views.

## Accessibility and Dropdown Behavior (Related UI)
- File: `ui/app.js`
- Sections: `toggleSortMenu`, `toggleEyeFocusMenu`, `bindDropdowns`, `handleGlobalDropdownClose`
- Changes:
  - Added `aria-expanded` and keyboard interactions (Enter/Space) for triggers.
  - Implemented outside-click and Escape-to-close handlers for menus.
- Rationale: Robust, accessible dropdown behavior across browsers.

## Validation Notes
- Verified selectors and grid transitions remain intact.
- No positional changes to existing elements; updates preserve current layout.
- Next step: Visual QA via preview to confirm responsiveness across viewports.