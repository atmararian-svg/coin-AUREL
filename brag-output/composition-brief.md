# Hyperframes Composition Brief: Arian Atmar — Portfolio

## Objective
Create a short launch-style brag video for arianatmar.com.

## Output
- Composition directory: `brag-output/composition/`
- Rendered video: `brag-output/brag.mp4`
- Format: landscape — 1920x1080, 30fps
- Duration: 23 seconds

## Source Material
- Project root: live site https://www.arianatmar.com (no local source code; material captured with headless Chromium into `brag-output/work/`)
- Primary files read: `/` HTML + CSS chunks, `/about` text, `/all-works` screenshot, project-card images `/img/project-cards/*.webp`
- Product name: Arian Atmar
- Tagline / strongest claim: "An Award-Winning Senior UI/UX Designer working with studios around the World."
- Key UI or visual moment to recreate: the "Arian ——— Atmar" line intro and the homepage "selected works" six-card grid
- Copy that must appear verbatim:
  - An Award-Winning / Senior UI/UX Designer
  - 001 Razed Mods '26 · 002 SŌM Power '26 · 003 SŌM '26 · 004 TPL '26 · 005 Kanoi '25 · 006 The Luxe Week '24
  - 12+ AWARDS
  - Awwwards · Site of the Day / FWA · FWA of the Day / CSSDA · Website of the Day
  - 6+ years / 30+ major projects
  - Specializing in UI/UX & web design / arianatmar.com

## Creative Direction
- Tone preset: app-store
- Creative direction: Numtera-style kinetic-type launch film (dynamic text movement) applied to a minimalist award-winning designer portfolio
- Interpretation: premium and clean, type always in motion (blur-in, scale-down, masked rise, slam, side-snap, gradient sweep), fast entrances with honest holds, a single accent hue.
- Angle: a designer portfolio launched like a product, using the site's own design language.
- Hook: "Meet" giant gradient word → site's "Arian ——— Atmar" intro.
- Outro / punchline: "Arian ——— Atmar" lockup + arianatmar.com.
- Avoid: generic SaaS language, abstract filler, redesigning the brand; no personal photo, no email.

## Visual Identity
- Background: #ffffff (dark flip #0d0d0d for recognition beat)
- Text: #0d0d0d; secondary grey darkened to pass contrast
- Accent: #ff2f00
- Display font: Inter (variable, shipped locally)
- Body font: Inter
- Visual references from the project: line intro, selected-works grid + faint giant "selected works" wordmark, accent-highlighted words on /about

## Storyboard
Use the storyboard in `brag-output/brag-plan.md` as the creative contract.

1. Hook — 2.5s — "Meet" → "Arian ——— Atmar"
2. The claim — 4.0s — "An Award-Winning / Senior UI/UX Designer"
3. Selected works — 5.5s — six real cards dealt on beats, match-cut into SŌM Power
4. Recognition — 4.0s — "12+ AWARDS" count-up + three award lines
5. Numbers — 3.0s — "6+ years", "30+ major projects"
6. Lockup — 4.0s — "Arian ——— Atmar", tagline, URL

## Audio
- Audio role: warm upbeat bed + light SFX layer
- Audio arc: bed from frame 0, accents follow the kinetic type, two soft impacts on stats, bell on lockup, bed fades under the hold
- Music: `assets/music/bed.mp3` (vol-1 trimmed to 23s with a baked 1.6s fade-out)
- Music cue guidance: bundled preset `.claude/skills/brag/assets/music/cues/happy-beats-business-moves-vol-1-by-ende-dot-app.music-cues.json` — 120.19 BPM; strong locks 16.02, 17.02, 20.02; beat grid on x.02 / x.52
- Audio-reactive treatment: subtle; RMS/bass drive a soft #ff2f00 radial glow (opacity/scale)
- SFX: drop_001, ui/click2, interface/click_003, casino/card-slide-1, impactSoft_medium_001/004, impactBell_heavy_000 at 0.55–0.75
