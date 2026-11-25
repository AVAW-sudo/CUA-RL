"""Add final checkpoint durability without changing the pinned trainer's math."""


def add_final_save(source):
    start = '        if global_step % save_interval == 0:\n'
    end = '        ray.get(actor.step_finished())\n'
    empty_end = '            global_step += 1\n            continue\n'
    if any(source.count(marker) != 1 for marker in [start, end, empty_end]):
        raise ValueError('Unknown checkpoint or empty-batch boundary in the pinned trainer')
    block = source[source.index(start):source.index(end)]
    replacement = block.replace(start, '        if global_step % save_interval == 0 or global_step == max_steps - 1:\n', 1)
    save_call = '            actor.save_checkpoint(str(checkpoint_dir))\n'
    if replacement.count(save_call) != 1:
        raise ValueError('Unknown collective checkpoint call in the pinned trainer')
    # CollectiveHandle returns one ObjectRef per actor. Publish the iteration
    # marker only after every model, optimizer and HF save has succeeded.
    replacement = replacement.replace(save_call, '            ray.get(actor.save_checkpoint(str(checkpoint_dir)))\n')
    marker = ('            with open(save_path / "latest_checkpointed_iteration.txt", "w") as f:\n'
              '                f.write(str(global_step))\n')
    if marker in replacement:
        replacement = replacement.replace(marker,
            '            marker_tmp = save_path / "latest_checkpointed_iteration.tmp"\n'
            '            marker_tmp.write_text(str(global_step))\n'
            '            marker_tmp.replace(save_path / "latest_checkpointed_iteration.txt")\n')
    # The original empty-batch branch skips the normal save block. Save the
    # unchanged actor and sampler state there as well so a final empty batch
    # cannot discard all preceding updates. Metrics still identify the skip.
    empty_save = ''.join('    ' + line if line.strip() else line
                         for line in replacement.splitlines(keepends=True))
    return source.replace(block, replacement, 1).replace(empty_end, empty_save + empty_end, 1)
