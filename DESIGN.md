---
name: CareerPilot AI, printed career report
description: Warm off-white paper, black ink, Newsreader headings and figures, and one Prussian-blue accent that is used only for the resume score figure, the confidence figure and the career-match bars. The results page reads like a printed report, not a dashboard.
source_of_truth: static/css/theme.css (tokens) and static/css/app.css (components)

colors:
  light:
    bg: "#f7f5f0"            # page ground, warm paper
    panel: "#fcfbf8"         # input card, loading panel
    panel-2: "#f2efe9"       # one tone step down: chips, drop zone
    fill: "#ebe8e2"          # bar and ring tracks
    border: "#e5e3df"        # hairline rules
    border-strong: "#c8c4be" # dashed "missing" chips, step markers
    border-control: "#8a857b" # outlines of form controls and buttons (3:1)
    rule: "#1a1917"          # the rule above each report section
    text: "#1a1917"          # ink, 16:1 on paper
    text-2: "#55524b"        # secondary, 7.2:1
    text-3: "#6e6a62"        # muted, 4.9:1
    ink: "#1a1917"           # primary buttons
    accent: "#1e4d78"        # Prussian blue, 8.1:1
    score-strong: "#2d6a3e"
    score-needs-work: "#8a5a00"
    score-weak: "#b0281f"
  dark:
    bg: "#171614"
    panel: "#1f1e1b"
    panel-2: "#262521"
    fill: "#2c2a26"
    border: "#34322e"
    border-strong: "#4a4741"
    border-control: "#77726a"
    rule: "#b8b3a8"          # softer than full ink, which glares on dark
    text: "#eeebe4"
    text-2: "#b8b3a8"
    text-3: "#9a958a"
    ink: "#eeebe4"
    accent: "#8db6dd"
    score-strong: "#7cc08c"
    score-needs-work: "#d9a640"
    score-weak: "#ef8a7f"

typography:
  display: { family: "Newsreader (Google Fonts, OFL)", weight: 500, use: "headings and big figures" }
  text: { family: "system-ui (Segoe UI, San Francisco, Roboto)", use: "reading text and controls" }
  scale:            # rem, so the browser text-size setting applies; px at the default 16px
    hero: { desktop: "3.75rem (60px, 56px beside the input card)", phone: "2.375rem (38px)", line-height: 1.04, tracking: "-0.025em" }
    h1-figures: { desktop: "2.5rem (40px)", phone: "1.875rem (30px)" }
    h3-sections: { desktop: "1.5rem (24px)", phone: "1.375rem (22px)", line-height: 1.2 }
    lg: "1.125rem (18px)"
    base: "1rem (16px), line-height 1.55"
    sm: "0.875rem (14px)"
    xs: "0.8125rem (13px)"
  figures: "lining-nums tabular-nums on every number"

rounded:
  "0": "bars, report sections, job rows, tables"
  sm: "4px: chips, tags, notes, tooltip"
  md: "8px: buttons, inputs, textarea, drop zone"
  lg: "12px: the input card and the loading panel only"
  round: "50%: true circles only (score ring, roadmap step markers, tip numbers)"
  pills: none

spacing: "4px grid: 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80"
sizes: { tap-target: 44px, control-height: 40px, header: "60px (56px on phones)", page-max: 1120px }
motion: { fast: 120ms, sheet: 180ms, ease-out: "cubic-bezier(0.22, 1, 0.36, 1)", entrance: "240ms rise of 6px, 40ms stagger", press: "scale(0.97), 160ms" }
---

# CareerPilot AI design

The app should feel like a career report printed on good paper. Headings and figures are set in Newsreader, reading text in the system font, everything in black ink on warm paper. Sections are separated by rules, not boxes. The tokens live in `static/css/theme.css`; `app.css` uses only those variables.

## The accent rule

The Prussian-blue accent appears in exactly three places:

1. The resume score figure (the number inside the score ring).
2. The confidence figure and its bar in the summary.
3. The career-match bars (the top match in full blue, the runners-up in a lighter mix of the same blue).

Everything else is ink: buttons, links, focus rings, tabs, the drop zone, the loading state, the current roadmap step, selected what-if chips and the "Why this match" words (ink semibold on a paper chip). If a new element seems to need colour, use weight or size first.

The green, amber and red score colours are not accents. They mean strong, needs work and weak, and appear only on the score ring, the score breakdown bars, skill chips you already have, roadmap steps you have done, error states, and the changes in "Compare with another version" (green = better, red = worse). A change is never colour alone: it keeps its sign or word (+3, −2, up, weaker), and a change that is neither better nor worse (a different 2nd match) stays ink. `tests/test_ui.py` checks the ring, breakdown and change colours.

## Layout

- **Home:** a left-aligned Newsreader headline with the subtitle and facts, beside the input card (one column on phones). Below it, "What's in your report" is a ruled contents list of four items, with no cards and no icons.
- **Results:** a masthead with the "Career report" title, what was analyzed and the three actions. A summary row of four figures sits between a 2px ink rule and a hairline. Then the sections, as a two-column spread on desktop (careers and gap, score and roadmap, courses and what-if, then jobs across the full width) and one column on phones. The DOM order is the reading order; `grid-template-areas` place them on desktop.
- Breakpoints: 560px (phone tweaks), 760px (type scale steps down), 1024px (two columns).

## Components

| Component | Rules |
|---|---|
| Report section | No box: `border-top: 1px solid var(--rule)`, 20px above the heading. Heading is an h2 in Newsreader, with a 14px subtitle in `text-2`. Sub-headings ("You have", "Tips to improve") are h3 in the system font, 14px semibold, sentence case. |
| Summary figures | 40px Newsreader, tabular figures. The confidence figure is the accent; the career name is 32px and may wrap. Thin vertical rules between figures, with 16-24px of space beside each rule. |
| Buttons | Primary: ink fill, paper text. Secondary: paper fill, `border-control` outline. 8px radius, 40px tall (44px on touch screens), press scales to 0.97. No shadows. |
| Links | Ink, underlined with a 1px `border-strong` line that darkens to ink on hover. |
| Tabs | Underline tabs: the selected one has a 2px ink rule. 44px tall. |
| Input card | The only floating card: `panel` fill, hairline border, 12px radius, no shadow. |
| Drop zone | Dashed `border-control` outline, `panel-2` fill, 8px radius. The upload icon sits on its own, with no tile. |
| Chips | 4px radius. Default: `panel-2` fill and a hairline. Have: green tint with a check. Missing: dashed outline. Why this match: ink semibold on `panel-2`. Single letters ("R") keep a 44px minimum width. |
| Bars | Native `<progress>`, 6px, square ends. Default value colour is `text-2`. Accent only on career-match and confidence bars; level colours on the score breakdown. |
| Score ring | SVG circle in the level colour with square ends; the figure inside is the accent. |
| Roadmap | A vertical timeline with circular step markers: green filled when a stage is complete, an ink outline for the current stage. |
| Long lists | Ten items, then a `<details>` with a 44px "Show all N" / "Show fewer" summary (works without JavaScript). |
| Jobs | Ruled listing rows, not boxes: a Newsreader title link, "Company · Location", then "View job". |
| Tooltip | The one element with a shadow, because it floats. Inverted ink, 8px radius. |
| Logo | A compass needle in ink with no tile; it uses `currentColor`. The favicon switches to paper colour in dark mode; the PNG icons put the mark on a paper square. |

## Do

- Use the tokens; never hard-code a colour in `app.css`.
- Keep figures in Newsreader with tabular numbers.
- Separate content with rules and space; make the most important thing the largest.
- Give every control a 44px target on touch screens and a visible ink focus ring.
- Check light and dark, at 390px and 1440px, before calling a change done.

## Don't

- Don't use the blue for anything except the three places above.
- Don't add cards, icon tiles, eyebrow labels, pills, gradients or shadows on things at rest.
- Don't use emoji or Unicode symbols as icons; use `icon()` (Lucide) from `templates/partials/macros.html`.
- Don't add inline styles or inline JavaScript handlers (`tests/test_ui.py` checks).
