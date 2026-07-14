def build_skills_system_prompt(
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
    compact_categories: "frozenset[str] | None" = None,
    query_text: "str | None" = None,
) -> str:
    """Build a compact skill index for the system prompt.

    Two-layer cache:
      1. In-process LRU dict keyed by (skills_dir, tools, toolsets, hidden,
         query_prefix)
      2. Disk snapshot (``.skills_prompt_snapshot.json``) validated by
         mtime/size manifest — survives process restarts

    Falls back to a full filesystem scan when both layers miss.

    External skill directories (``skills.external_dirs`` in config.yaml) are
    scanned alongside the local ``~/.hermes/skills/`` directory.  External dirs
    are read-only — they appear in the index but new skills are always created
    in the local dir.  Local skills take precedence when names collide.

    ``compact_categories`` (e.g. from the coding posture — see
    agent/coding_context.py) demotes whole categories to a names-only line in
    the rendered index. Nothing is ever hidden: every skill name stays
    visible and loadable via ``skill_view`` / ``skills_list``; only the
    descriptions are dropped, and a footer note explains the demotion.

    ``query_text`` (the user's current message) activates per-message hybrid
    skill retrieval when ``skills.semantic_search.enabled`` is true (the
    default).  Only the top-k most relevant skills are shown with full
    descriptions; all others appear as names-only so the agent can still load
    any of them via ``skill_view(name)``.  Falls back to full-index rendering
    on any error or when the skill count is at or below top_k.
    """
    skills_dir = get_skills_dir()
    external_dirs = get_all_skills_dirs()[1:]  # skip local (index 0)

    if not skills_dir.exists() and not external_dirs:
        return ""

    # ── Layer 1: in-process LRU cache ─────────────────────────────────
    # Include the resolved platform so per-platform disabled-skill lists
    # produce distinct cache entries (gateway serves multiple platforms).
    # query_text[:128] is added so per-query ranked results are cached
    # separately (recent queries stay warm within the 8-slot LRU window).
    _platform_hint = _current_session_platform_hint()

    disabled = get_disabled_skill_names(_platform_hint or None)
    cache_key = (
        str(skills_dir),
        tuple(str(d) for d in external_dirs),
        tuple(sorted(str(t) for t in (available_tools or set()))),
        tuple(sorted(str(ts) for ts in (available_toolsets or set()))),
        _platform_hint,
        tuple(sorted(disabled)),
        tuple(sorted(compact_categories or ())),
        (query_text or "")[:128],
    )
    with _SKILLS_PROMPT_CACHE_LOCK:
        cached = _SKILLS_PROMPT_CACHE.get(cache_key)
        if cached is not None:
            _SKILLS_PROMPT_CACHE.move_to_end(cache_key)
            return cached

    # ── Layer 2: disk snapshot ────────────────────────────────────────
    snapshot = _load_skills_snapshot(skills_dir)

    skills_by_category: dict[str, list[tuple[str, str]]] = {}
    category_descriptions: dict[str, str] = {}

    if snapshot is not None:
        # Fast path: use pre-parsed metadata from disk
        for entry in snapshot.get("skills", []):
            if not isinstance(entry, dict):
                continue
            skill_name = entry.get("skill_name") or ""
            category = entry.get("category") or "general"
            frontmatter_name = entry.get("frontmatter_name") or skill_name
            platforms = entry.get("platforms") or []
            if not skill_matches_platform_list(platforms):
                continue
            if frontmatter_name in disabled or skill_name in disabled:
                continue
            if not _skill_should_show(
                entry.get("conditions") or {},
                available_tools,
                available_toolsets,
            ):
                continue
            skills_by_category.setdefault(category, []).append(
                (frontmatter_name, entry.get("description", ""))
            )
        category_descriptions = {
            str(k): str(v)
            for k, v in (snapshot.get("category_descriptions") or {}).items()
        }
    else:
        # Cold path: full filesystem scan + write snapshot for next time
        skill_entries: list[dict] = []
        for skill_file in iter_skill_index_files(skills_dir, "SKILL.md"):
            is_compatible, frontmatter, desc = _parse_skill_file(skill_file)
            entry = _build_snapshot_entry(skill_file, skills_dir, frontmatter, desc)
            skill_entries.append(entry)
            if not is_compatible:
                continue
            skill_name = entry["skill_name"]
            if entry["frontmatter_name"] in disabled or skill_name in disabled:
                continue
            if not _skill_should_show(
                extract_skill_conditions(frontmatter),
                available_tools,
                available_toolsets,
            ):
                continue
            skills_by_category.setdefault(entry["category"], []).append(
                (entry["frontmatter_name"], entry["description"])
            )

        # Read category-level DESCRIPTION.md files
        for desc_file in iter_skill_index_files(skills_dir, "DESCRIPTION.md"):
            try:
                content = desc_file.read_text(encoding="utf-8")
                fm, _ = parse_frontmatter(content)
                cat_desc = fm.get("description")
                if not cat_desc:
                    continue
                rel = desc_file.relative_to(skills_dir)
                cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
                category_descriptions[cat] = str(cat_desc).strip().strip("'\"")
            except Exception as e:
                logger.debug("Could not read skill description %s: %s", desc_file, e)

        _write_skills_snapshot(
            skills_dir,
            _build_skills_manifest(skills_dir),
            skill_entries,
            category_descriptions,
        )

    # ── External skill directories ─────────────────────────────────────
    # Scan external dirs directly (no snapshot caching — they're read-only
    # and typically small).  Local skills already in skills_by_category take
    # precedence: we track seen names and skip duplicates from external dirs.
    seen_skill_names: set[str] = set()
    for cat_skills in skills_by_category.values():
        for name, _desc in cat_skills:
            seen_skill_names.add(name)

    for ext_dir in external_dirs:
        if not ext_dir.exists():
            continue
        for skill_file in iter_skill_index_files(ext_dir, "SKILL.md"):
            try:
                is_compatible, frontmatter, desc = _parse_skill_file(skill_file)
                if not is_compatible:
                    continue
                entry = _build_snapshot_entry(skill_file, ext_dir, frontmatter, desc)
                skill_name = entry["skill_name"]
                frontmatter_name = entry["frontmatter_name"]
                if frontmatter_name in seen_skill_names:
                    continue
                if frontmatter_name in disabled or skill_name in disabled:
                    continue
                if not _skill_should_show(
                    extract_skill_conditions(frontmatter),
                    available_tools,
                    available_toolsets,
                ):
                    continue
                seen_skill_names.add(frontmatter_name)
                skills_by_category.setdefault(entry["category"], []).append(
                    (frontmatter_name, entry["description"])
                )
            except Exception as e:
                logger.debug("Error reading external skill %s: %s", skill_file, e)

        # External category descriptions
        for desc_file in iter_skill_index_files(ext_dir, "DESCRIPTION.md"):
            try:
                content = desc_file.read_text(encoding="utf-8")
                fm, _ = parse_frontmatter(content)
                cat_desc = fm.get("description")
                if not cat_desc:
                    continue
                rel = desc_file.relative_to(ext_dir)
                cat = "/".join(rel.parts[:-1]) if len(rel.parts) > 1 else "general"
                category_descriptions.setdefault(cat, str(cat_desc).strip().strip("'\""))
            except Exception as e:
                logger.debug("Could not read external skill description %s: %s", desc_file, e)

    # Posture-driven category demotion (e.g. non-coding skills while pairing
    # on code). Demoted categories stay in the index as a single names-only
    # line — descriptions are dropped to cut noise, but every skill name
    # remains visible so memory-anchored recall ("load <name>") keeps working.
    # NEVER remove entries entirely: agent-created skills are the model's
    # project memory, and models don't reach for skills_list to rediscover
    # what the index stops showing them. Match on the top-level category
    # segment so nested categories ("social-media/twitter") are demoted with
    # their parent.
    demoted = frozenset(
        cat for cat in skills_by_category
        if cat.split("/", 1)[0] in (compact_categories or frozenset())
    )

    hidden_note = ""
    if demoted:
        hidden_note = (
            "\n(Categories marked [names only] are outside the current coding "
            "context, so their descriptions are omitted — the skills work "
            "normally and load with skill_view(name) as usual.)"
        )

    # ── Hybrid per-message skill retrieval ────────────────────────────
    # When query_text is present and semantic_search.enabled is true (the
    # default), rank all skills against the user's message using BM25 +
    # optional dense re-ranking fused with RRF. Split the result into two
    # tiers:
    #   featured_skills  → rendered with full "name: description" lines
    #   everyone else    → rendered as a names-only footer line
    #
    # The agent can still call skill_view(name) on any non-featured skill;
    # no information is ever suppressed, only condensed.
    #
    # Falls back to full-index rendering (featured_skills = None) on any
    # error, when retrieval is disabled, or when skill count ≤ top_k.
    featured_skills: "set[str] | None" = None
    if query_text:
        try:
            from hermes_cli.config import load_config as _load_config
            _cfg = _load_config()
            _ss: dict = ((_cfg.get("skills") or {}).get("semantic_search") or {})
            if _ss.get("enabled", True):
                _top_k = max(1, int(_ss.get("top_k", 5)))
                _all_skills = [
                    {"name": name, "description": desc}
                    for cat_skills in skills_by_category.values()
                    for name, desc in cat_skills
                ]
                if len(_all_skills) > _top_k:
                    from agent.skill_retrieval import retrieve_skills
                    featured_skills = retrieve_skills(
                        query_text,
                        _all_skills,
                        top_k=_top_k,
                        embedding_cfg=_ss if _ss.get("embedding_model") else None,
                    )
        except Exception as _exc:
            logger.debug(
                "Skill retrieval failed, falling back to full index: %s", _exc
            )
            featured_skills = None

    if not skills_by_category:
        result = ""
    else:
        index_lines: list[str] = []
        non_featured_names: list[str] = []

        for category in sorted(skills_by_category.keys()):
            # Deduplicate and sort skills within each category
            seen: set[str] = set()
            if category in demoted:
                names = sorted({name for name, _ in skills_by_category[category]})
                index_lines.append(f"  {category} [names only]: {', '.join(names)}")
                continue
            cat_desc = category_descriptions.get(category, "")
            # Track whether any featured skill lands in this category so we
            # can remove an orphaned category header.
            cat_header_idx = len(index_lines)
            if cat_desc:
                index_lines.append(f"  {category}: {cat_desc}")
            else:
                index_lines.append(f"  {category}:")
            has_featured_in_category = False
            for name, desc in sorted(skills_by_category[category], key=lambda x: x[0]):
                if name in seen:
                    continue
                seen.add(name)
                # Retrieval ran: demote non-featured to the names-only bucket
                if featured_skills is not None and name not in featured_skills:
                    non_featured_names.append(name)
                    continue
                has_featured_in_category = True
                if desc:
                    index_lines.append(f"    - {name}: {desc}")
                else:
                    index_lines.append(f"    - {name}")
            # If retrieval ran and demoted every skill in this category,
            # remove the orphaned category header we just added.
            if featured_skills is not None and not has_featured_in_category:
                del index_lines[cat_header_idx]

        # Non-featured skills: names-only footer so the agent can reach them
        if non_featured_names:
            index_lines.append(
                "  [other skills, names only]: "
                + ", ".join(sorted(non_featured_names))
            )
            index_lines.append(
                "  (load any with skill_view(name) — only the most relevant "
                "skills for this turn are shown with full descriptions above)"
            )

        result = (
            "## Skills (mandatory)\n"
            "Before replying, scan the skills below. If a skill matches or is even partially relevant "
            "to your task, you MUST load it with skill_view(name) and follow its instructions. "
            "Err on the side of loading — it is always better to have context you don't need "
            "than to miss critical steps, pitfalls, or established workflows. "
            "Skills contain specialized knowledge — API endpoints, tool-specific commands, "
            "and proven workflows that outperform general-purpose approaches. Load the skill "
            "even if you think you could handle the task with basic tools like web_search or terminal. "
            "Skills also encode the user's preferred approach, conventions, and quality standards "
            "for tasks like code review, planning, and testing — load them even for tasks you "
            "already know how to do, because the skill defines how it should be done here.\n"
            "Whenever the user asks you to configure, set up, install, enable, disable, modify, "
            "or troubleshoot Hermes Agent itself — its CLI, config, models, providers, tools, "
            "skills, voice, gateway, plugins, or any feature — load the `hermes-agent` skill "
            "first. It has the actual commands (e.g. `hermes config set …`, `hermes tools`, "
            "`hermes setup`) so you don't have to guess or invent workarounds.\n"
            "If a skill has issues, fix it with skill_manage(action='patch').\n"
            "After difficult/iterative tasks, offer to save as a skill. "
            "If a skill you loaded was missing steps, had wrong commands, or needed "
            "pitfalls you discovered, update it before finishing.\n"
            "\n"
            "<available_skills>\n"
            + "\n".join(index_lines) + "\n"
            "</available_skills>\n"
            "\n"
            "Only proceed without loading a skill if genuinely none are relevant to the task."
            + hidden_note
        )

    # ── Store in LRU cache ────────────────────────────────────────────
    with _SKILLS_PROMPT_CACHE_LOCK:
        _SKILLS_PROMPT_CACHE[cache_key] = result
        _SKILLS_PROMPT_CACHE.move_to_end(cache_key)
        while len(_SKILLS_PROMPT_CACHE) > _SKILLS_PROMPT_CACHE_MAX:
            _SKILLS_PROMPT_CACHE.popitem(last=False)

    return result
