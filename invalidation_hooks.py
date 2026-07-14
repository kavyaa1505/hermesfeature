"""
Invalidation hook snippets for skill mutation points.

These are drop-in snippets to add after each write/delete/install/remove
operation in the files listed below.  Each is wrapped in a bare try/except
so a broken index never propagates into a skill management failure.

─────────────────────────────────────────────────────────────────────────────
tools/skill_manager_tool.py
─────────────────────────────────────────────────────────────────────────────
Add after any create / patch / delete operation completes successfully.
Replace ``skill_name`` with the actual name variable used in context.

    # Invalidate the semantic skill index so the next turn re-ranks from fresh.
    try:
        from agent.skill_retrieval import get_index
        get_index().invalidate([skill_name])
    except Exception:
        pass  # index is best-effort; never block skill management

─────────────────────────────────────────────────────────────────────────────
tools/skills_sync.py  — end of sync_skills()
─────────────────────────────────────────────────────────────────────────────
Add at the end of sync_skills() after bundled skills are seeded/updated:

    # Full index invalidation: bundled skills may have changed.
    # The background build on the next turn will re-embed all stale entries.
    try:
        from agent.skill_retrieval import get_index
        get_index().invalidate()
    except Exception:
        pass

─────────────────────────────────────────────────────────────────────────────
hermes_cli/skills_hub.py  — do_install() and do_remove()
─────────────────────────────────────────────────────────────────────────────
Add at the end of do_install() after the skill is written to disk:

    if invalidate_cache:
        try:
            from agent.skill_retrieval import get_index
            get_index().invalidate([skill_name])  # partial invalidation
        except Exception:
            pass

Add at the end of do_remove() after the skill directory is deleted:

    try:
        from agent.skill_retrieval import get_index
        get_index().invalidate([name])  # name = the removed skill's name
    except Exception:
        pass
"""
