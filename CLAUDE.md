> Part of [[BRAIN-INDEX]]

# Vibemesh2 - Agent Instructions

## What Is This Brain?
This brain tracks the Vibemesh2 research project: a MuJoCo-based simulation environment for audio-based slip recovery using a LEAP Hand on an xArm manipulator. The goal is to build a sim-validated RL policy that detects object slip and executes corrective finger actions, eventually transferring to real hardware.

## Owner
- **Role**: Benton (Zyfex) — lead implementer
- **Context**: MuJoCo setup is underway. YCB objects are stable in sim. Currently tuning friction so the LEAP hand can pick up objects. Uksang is the faculty advisor/mentor.
- **Goals**: Complete object grasping, then train a reactive RL slip recovery policy ready for hardware transfer.

## Brain Structure
- [[Simulation]] - MuJoCo env config, physics params, rendering
- [[Robot]] - xArm + LEAP Hand models, kinematics, actuation
- [[Objects]] - YCB/HOPE mesh library, per-object physics params
- [[Policy]] - RL slip recovery design, reward, training, eval
- [[Research]] - Papers, references, theory
- [[Experiments]] - Experiment logs, results, ablations
- [[Handoffs]] - Session continuity notes
- [[Templates]] - Reusable note structures

## Conventions
- Use [[wikilinks]] for all cross-references between notes, but ONLY link to files that exist.
- Keep files concise and actionable.
- Check [[Assets]] for related diagrams or screenshots when working on any task.
- Update Handoffs/ at the end of every work session.
- Reference the [[Execution-Plan]] as the source of truth for phase order.

## Assets
The [[Assets]] folder holds images, videos, PDFs, and other media. When working on any task, check Assets/ for related materials. You can analyze images, read PDFs, and process any file dropped there.

## Agent Personas
Available specialized agents in .claude/agents/:
- [[builder]] - Implements simulation code, scripts, XML configs
- [[researcher]] - Finds papers, synthesizes theory, tracks references
- [[rl-engineer]] - Designs RL policy, reward functions, training pipelines
- [[sim-engineer]] - Debugs physics, contact params, mesh quality issues

## Commands
- /init-braintree - Initialize a new brain
- /resume-braintree - Resume from where you left off
- /wrap-up-braintree - End session with proper handoff
- /status-braintree - View progress dashboard
- /plan-braintree [step] - Plan a specific step
- /sprint-braintree - Plan the week's work
- /sync-braintree - Health check and sync
- /feature-braintree [name] - Plan a new feature
