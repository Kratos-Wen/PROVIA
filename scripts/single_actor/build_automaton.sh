#!/usr/bin/env bash
# Procedure languages from correct training executions, training examples, prefix evidence, step effects and the
# predictive-state automata. Usage: build_automaton.sh DATASET
source scripts/common.sh
resolve_dataset "$1"
observer=$run/observer/best.pt
python -m provia.automaton.compile_languages --manifest "$manifest" --feature-root "$features" \
  --checkpoint "$observer" --output-dir "$run/languages" --fold 0 --device "$device"
python -m provia.training.prepare_single_actor_examples --manifest "$manifest" --feature-root "$features" \
  --observer-checkpoint "$observer" --language-registry "$run/languages/registry.json" \
  --output "$run/examples.pt" --fold 0 --device "$device"
python -m provia.automaton.prefix_evidence_single_actor --examples "$run/examples.pt" \
  --output-dir "$run/prefix_evidence" --device "$device"
python -m provia.automaton.step_effects completion --examples "$run/examples.pt" \
  --language-registry "$run/languages/registry.json" --output-dir "$run/step_effects"
python -m provia.automaton.compile_predictive_automata --examples "$run/examples.pt" \
  --source-registry "$run/languages/registry.json" --output-dir "$run/automaton"
