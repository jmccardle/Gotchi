# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Gotchi is a virtual pet/Tamagotchi-style game that evaluates how LLMs behave as caregivers. The project tests sustained attention, latent rule inference, and intrinsic motivation through incomplete information gameplay.

## Commands

### Running the Game
```bash
# Interactive gameplay
python gotchi.py

# Automated LLM experiments
python logits_gotchi.py       # Logit-based action selection
python logits_cot_gotchi.py   # Chain-of-thought reasoning
python dspy_gotchi.py         # DSPy framework implementation
```

## Architecture

### Core Game Engine (gotchi.py)
- Event-driven simulation with 120-second stat decay intervals
- Non-blocking input via threading
- ANSI escape codes for partial screen updates
- State machine: normal → sick/away → dead

### LLM Implementations
1. **logits_gotchi.py**: Uses logit biases to select actions probabilistically
2. **logits_cot_gotchi.py**: Adds reasoning step before action selection
3. **dspy_gotchi.py**: Structured prompting with DSPy framework

### Key Patterns
- All LLM variants communicate with local server at `http://192.168.1.100:8853/v1`
- Game state tracked via dictionary: `{hunger, happiness, energy, friendship, sick, away}`
- Actions mapped to keys: Feed [F], Play [P], Sleep [S], Quit [Q]
- Hidden mechanics revealed through experimentation (weather, needs phrases, wandering)

## Important Mechanics
- **Visible stats**: Hunger, Happiness, Energy (0-5 range)
- **Hidden stat**: Friendship (affects pet wandering behavior)
- **Failure conditions**: Any stat reaching 0 or friendship depletion
- **Special events**: Random weather, need phrases grant bonuses
- **Sickness**: Triggered by weather or low stats, requires multiple actions to cure