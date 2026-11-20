"""Add final checkpoint durability without changing the pinned trainer's math."""


def add_final_save(source):
    start = '        if global_step % save_interval == 0:\n'
    end = '        ray.get(actor.step_finished())\n'
    empty_end = '            global_step += 1\n            continue\n'
    if any(source.count(marker) != 1 for marker in [start, end, empty_end]):
        raise ValueError('Unknown checkpoint or empty-batch boundary in the pinned trainer')
    block = source[source.index(start):source.index(end)]
    replacement = block.replace(start, '        if global_step % save_interval == 0 or global_step == max_steps - 1:\n', 1)
    # The original empty-batch branch skips the normal save block. Save the
    # unchanged actor and sampler state there as well so a final empty batch
    # cannot discard all preceding updates. Metrics still identify the skip.
    empty_save = ''.join('    ' + line if line.strip() else line
                         for line in replacement.splitlines(keepends=True))
    return source.replace(block, replacement, 1).replace(empty_end, empty_save + empty_end, 1)
