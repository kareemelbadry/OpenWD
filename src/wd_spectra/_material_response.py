"""Admissible finite-difference stencils for the same material equations."""
from .nonlinear import RecoverableEvaluationError


def temperature_response_probes(evaluate_offset, step, *, centered):
    """Return primary, optional opposite probe, and signed primary ln-T step.

    ``evaluate_offset`` evaluates actual materials at a uniform log-temperature
    offset from an already validated base state. An invalid physical probe
    chooses the other side, never another EOS. Programming errors propagate.
    If neither side is valid the domain exception propagates. Interior
    forward/centered stencils keep their original offsets and values exactly.
    """
    try:
        primary = evaluate_offset(step)
    except RecoverableEvaluationError:
        return evaluate_offset(-step), None, -step
    if not centered:
        return primary, None, step
    try:
        opposite = evaluate_offset(-step)
    except RecoverableEvaluationError:
        opposite = None
    return primary, opposite, step
