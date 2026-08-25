# Learn Installed Skills Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use the executing-plans skill to implement this plan task-by-task.

**Goal:** Create a comprehensive learning plan to understand and effectively use the installed skills (superpowers, claude-mem, awesome-claude-code) to learn from their respective repositories.

**Architecture:** This plan will systematically explore each skill set, starting with understanding their purpose and structure, then diving into specific skills within each collection, and finally practicing application of these skills to learn from the source repositories.

**Tech Stack:** Bash, Git, Claude Code skills system

---

### Task 1: Explore Superpowers Skill Collection

**Files:**
- Create: `docs/plans/superpowers-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:28-28`

**Step 1: Write failing test for superpowers understanding**

```bash
# Verify we can list superpowers skills
ls -la /home/elfakiris/.claude/skills/ | grep -E "(brainstorming|dispatching-parallel-agents|executing-plans|finishing-a-development-branch|receiving-code-review|requesting-code-review|subagent-driven-development|test-driven-development|using-git-worktrees|verification-before-completion|writing-plans|writing-skills|using-superpowers)"
```

Expected: FAIL with exit code 1 (if skills not verified)

**Step 2: Run test to verify it fails**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(brainstorming|dispatching-parallel-agents|executing-plans|finishing-a-development-branch|receiving-code-review|requesting-code-review|subagent-driven-development|test-driven-development|using-git-worktrees|verification-before-completion|writing-plans|writing-skills|using-superpowers)"`
Expected: Empty output or no matches

**Step 3: Write minimal implementation to verify superpowers skills**

```bash
# List superpowers skills to confirm installation
ls -la /home/elfakiris/.claude/skills/ | grep -E "(brainstorming|dispatching-parallel-agents|executing-plans|finishing-a-development-branch|receiving-code-review|requesting-code-review|subagent-driven-development|test-driven-development|using-git-worktrees|verification-before-completion|writing-plans|writing-skills|using-superpowers)"
```

**Step 4: Run test to verify it passes**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(brainstorming|dispatching-parallel-agents|executing-plans|finishing-a-development-branch|receiving-code-review|requesting-code-review|subagent-driven-development|test-driven-development|using-git-worktrees|verification-before-completion|writing-plans|writing-skills|using-superpowers)"`
Expected: List of matching skill directories

**Step 5: Commit**

```bash
git add docs/plans/2026-08-25-learn-installed-skills.md
git commit -m "feat: add superpowers skills verification task"
```

### Task 2: Create Superpowers Overview Document

**Files:**
- Create: `docs/plans/superpowers-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:60-60`

**Step 1: Write failing test for superpowers overview**

```bash
# Check if superpowers overview document exists
test -f /home/elfakiris/docs/plans/superpowers-overview.md
```

Expected: FAIL with exit code 1

**Step 2: Run test to verify it fails**

Run: `test -f /home/elfakiris/docs/plans/superpowers-overview.md`
Expected: Non-zero exit code

**Step 3: Write minimal implementation for superpowers overview**

```bash
# Create superpowers overview document
cat > /home/elfakiris/docs/plans/superpowers-overview.md << 'EOF'
# Superpowers Skills Overview

## Purpose
Superpowers is a complete software development methodology for coding agents, built on composable skills that guide brainstorming, planning, implementation, testing, review, and branch completion.

## Key Skills
- brainstorming: Explore user intent and design before implementation
- writing-plans: Create detailed implementation plans
- executing-plans: Execute plans with checkpoints
- test-driven-development: RED-GREEN-REFACTOR cycle
- requesting-code-review: Get feedback before merging
- receiving-code-review: Respond to code review feedback
- subagent-driven-development: Fast iteration with two-stage review
- using-git-worktrees: Isolated workspace for feature work
- finishing-a-development-branch: Merge/PR decision workflow
- verification-before-completion: Confirm work is actually done
- writing-skills: Create new skills following best practices
- dispatching-parallel-agents: Handle independent concurrent tasks
- systematic-debugging: 4-phase root cause process

## Usage
Skills are invoked automatically when relevant to the task. The using-superpowers skill establishes the foundation for skill invocation.

## References
- SKILL.md files in each skill directory
- https://github.com/obra/superpowers
EOF
```

**Step 4: Run test to verify it passes**

Run: `test -f /home/elfakiris/docs/plans/superpowers-overview.md`
Expected: Zero exit code (file exists)

**Step 5: Commit**

```bash
git add docs/plans/superpowers-overview.md
git commit -m "feat: create superpowers overview document"
```

### Task 3: Explore Claude-Mem Skill Collection

**Files:**
- Create: `docs/plans/claude-mem-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:92-92`

**Step 1: Write failing test for claude-mem understanding**

```bash
# Verify we can list claude-mem skills
ls -la /home/elfakiris/.claude/skills/ | grep -E "(knowledge-agent|learn-codebase)"
```

Expected: FAIL with exit code 1 (if skills not verified)

**Step 2: Run test to verify it fails**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(knowledge-agent|learn-codebase)"`
Expected: Empty output or no matches

**Step 3: Write minimal implementation to verify claude-mem skills**

```bash
# List claude-mem skills to confirm installation
ls -la /home/elfakiris/.claude/skills/ | grep -E "(knowledge-agent|learn-codebase)"
```

**Step 4: Run test to verify it passes**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(knowledge-agent|learn-codebase)"`
Expected: List of matching skill directories

**Step 5: Commit**

```bash
git add docs/plans/2026-08-25-learn-installed-skills.md
git commit -m "feat: add claude-mem skills verification task"
```

### Task 4: Create Claude-Mem Overview Document

**Files:**
- Create: `docs/plans/claude-mem-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:124-124`

**Step 1: Write failing test for claude-mem overview**

```bash
# Check if claude-mem overview document exists
test -f /home/elfakiris/docs/plans/claude-mem-overview.md
```

Expected: FAIL with exit code 1

**Step 2: Run test to verify it fails**

Run: `test -f /home/elfakiris/docs/plans/claude-mem-overview.md`
Expected: Non-zero exit code

**Step 3: Write minimal implementation for claude-mem overview**

```bash
# Create claude-mem overview document
cat > /home/elfakiris/docs/plans/claude-mem-overview.md << 'EOF'
# Claude-Mem Skills Overview

## Purpose
Claude-Mem provides skills for building AI-powered knowledge bases from observation history, enabling users to create focused "brains" from past work patterns and compile expertise on specific topics.

## Key Skills
- knowledge-agent: Build and query AI-powered knowledge bases from claude-mem observations
- learn-codebase: Prime a codebase by reading every source file in full
- smart-explore: Token-optimized code search
- timeline-report: Project history analysis
- weekly-digests: Per-week narrative chapters
- design-is: Design audit and analysis
- oh-my-issues: Issue clustering and analysis
- make-plan: Create plans from observations
- babysit: Monitor and shepherd GitHub pull requests
- pathfinder: Find files and code patterns
- memory-augmented-dev: Enhance development with memory
- version-bump: Automate version management
- mem-search: Search through memory stores

## Usage
These skills help users leverage their interaction history with Claude to build personalized knowledge systems and improve development efficiency.

## References
- SKILL.md files in each skill directory
- https://github.com/thedotmack/claude-mem
EOF
```

**Step 4: Run test to verify it passes**

Run: `test -f /home/elfakiris/docs/plans/claude-mem-overview.md`
Expected: Zero exit code (file exists)

**Step 5: Commit**

```bash
git add docs/plans/claude-mem-overview.md
git commit -m "feat: create claude-mem overview document"
```

### Task 5: Explore Awesome-Claude-Code Skill Collection

**Files:**
- Create: `docs/plans/awesome-claude-code-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:156-156`

**Step 1: Write failing test for awesome-claude-code understanding**

```bash
# Verify we can list awesome-claude-code skills
ls -la /home/elfakiris/.claude/skills/ | grep -E "(ui-ux-pro-max|banner-design|brand|design|design-system|slides|ui-styling)"
```

Expected: FAIL with exit code 1 (if skills not verified)

**Step 2: Run test to verify it fails**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(ui-ux-pro-max|banner-design|brand|design|design-system|slides|ui-styling)"`
Expected: Empty output or no matches

**Step 3: Write minimal implementation to verify awesome-claude-code skills**

```bash
# List awesome-claude-code skills to confirm installation
ls -la /home/elfakiris/.claude/skills/ | grep -E "(ui-ux-pro-max|banner-design|brand|design|design-system|slides|ui-styling)"
```

**Step 4: Run test to verify it passes**

Run: `ls -la /home/elfakiris/.claude/skills/ | grep -E "(ui-ux-pro-max|banner-design|brand|design|design-system|slides|ui-styling)"`
Expected: List of matching skill directories

**Step 5: Commit**

```bash
git add docs/plans/2026-08-25-learn-installed-skills.md
git commit -m "feat: add awesome-claude-code skills verification task"
```

### Task 6: Create Awesome-Claude-Code Overview Document

**Files:**
- Create: `docs/plans/awesome-claude-code-overview.md`
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:188-188`

**Step 1: Write failing test for awesome-claude-code overview**

```bash
# Check if awesome-claude-code overview document exists
test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md
```

Expected: FAIL with exit code 1

**Step 2: Run test to verify it fails**

Run: `test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md`
Expected: Non-zero exit code

**Step 3: Write minimal implementation for awesome-claude-code overview**

```bash
# Create awesome-claude-code overview document
cat > /home/elfakiris/docs/plans/awesome-claude-code-overview.md << 'EOF'
# Awesome-Claude-Code Skills Overview

## Purpose
Awesome-Claude-Code provides comprehensive UI/UX design intelligence for web, mobile, and desktop applications, offering design systems, component libraries, and styling guidance.

## Key Skills
- ui-ux-pro-max: UI/UX design intelligence with 50 styles, 21 palettes, 50 font pairings, 20 charts, 9 stacks
- banner-design: Design banners for social media, ads, website heroes
- brand: Brand voice, visual identity, messaging frameworks
- design: Comprehensive design skill including logo generation and corporate identity
- design-system: Token architecture and component specifications
- slides: Strategic HTML presentations with Chart.js and design tokens
- ui-styling: Beautiful, accessible UIs with shadcn/ui components and Tailwind CSS

## Usage
These skills provide AI-powered design assistance for creating professional interfaces across multiple platforms and frameworks.

## References
- SKILL.md files in each skill directory
- https://github.com/nextlevelbuilder/ui-ux-pro-max-skill
EOF
```

**Step 4: Run test to verify it passes**

Run: `test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md`
Expected: Zero exit code (file exists)

**Step 5: Commit**

```bash
git add docs/plans/awesome-claude-code-overview.md
git commit -m "feat: create awesome-claude-code overview document"
```

### Task 7: Practice Learning from Repositories

**Files:**
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:220-220`

**Step 1: Write failing test for repository learning practice**

```bash
# Verify we can access skill repositories
test -d /home/elfakiris/.claude/skills/brainstorming && test -d /home/elfakiris/.claude/skills/knowledge-agent && test -d /home/elfakiris/.claude/skills/ui-ux-pro-max
```

Expected: FAIL with exit code 1 (if directories don't exist)

**Step 2: Run test to verify it fails**

Run: `test -d /home/elfakiris/.claude/skills/brainstorming && test -d /home/elfakiris/.claude/skills/knowledge-agent && test -d /home/elfakiris/.claude/skills/ui-ux-pro-max`
Expected: Non-zero exit code

**Step 3: Write minimal implementation for repository learning practice**

```bash
# Verify skill repositories exist and have content
ls -la /home/elfakiris/.claude/skills/brainstorming/
ls -la /home/elfakiris/.claude/skills/knowledge-agent/
ls -la /home/elfakiris/.claude/skills/ui-ux-pro-max/
```

**Step 4: Run test to verify it passes**

Run: `ls -la /home/elfakiris/.claude/skills/brainstorming/ && ls -la /home/elfakiris/.claude/skills/knowledge-agent/ && ls -la /home/elfakiris/.claude/skills/ui-ux-pro-max/`
Expected: Directory listings showing SKILL.md and other files

**Step 5: Commit**

```bash
git add docs/plans/2026-08-25-learn-installed-skills.md
git commit -m "feat: add repository learning practice task"
```

### Task 8: Finalize Learning Plan Documentation

**Files:**
- Modify: `docs/plans/2026-08-25-learn-installed-skills.md:252-252`

**Step 1: Write failing test for final documentation**

```bash
# Verify all overview documents exist
test -f /home/elfakiris/docs/plans/superpowers-overview.md && test -f /home/elfakiris/docs/plans/claude-mem-overview.md && test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md
```

Expected: FAIL with exit code 1 (if any document missing)

**Step 2: Run test to verify it fails**

Run: `test -f /home/elfakiris/docs/plans/superpowers-overview.md && test -f /home/elfakiris/docs/plans/claude-mem-overview.md && test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md`
Expected: Non-zero exit code

**Step 3: Write minimal implementation for final documentation**

```bash
# Update main learning plan with completion note
echo -e "\n## Learning Completed\n\nAll skill overviews have been created and verified." >> /home/elfakiris/docs/plans/2026-08-25-learn-installed-skills.md
```

**Step 4: Run test to verify it passes**

Run: `test -f /home/elfakiris/docs/plans/superpowers-overview.md && test -f /home/elfakiris/docs/plans/claude-mem-overview.md && test -f /home/elfakiris/docs/plans/awesome-claude-code-overview.md`
Expected: Zero exit code

**Step 5: Commit**

```bash
git add docs/plans/2026-08-25-learn-installed-skills.md docs/plans/superpowers-overview.md docs/plans/claude-mem-overview.md docs/plans/awesome-claude-code-overview.md
git commit -m "feat: complete learning plan for installed skills"
```
## Learning Completed

All skill overviews have been created and verified.
