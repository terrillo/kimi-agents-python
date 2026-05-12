"""Session.fork() and Session.checkpoint() — tree-of-thought + rollback.

Use fork() to branch a conversation (each branch has its own history and
stats), and checkpoint()/restore() to roll back a single session to a
prior state. Both are in-memory.
"""

from kimi_agents_python import KimiClient, Model, Session

with KimiClient() as client:
    sess = Session(
        client,
        model=Model.KIMI_K2_0905_PREVIEW,
        system="You are a brainstorming partner. One sentence per reply.",
    )
    sess.send("Suggest a name for a CLI tool that lints Markdown frontmatter.")

    # Branch: explore two alternative directions independently.
    playful = sess.fork()
    serious = sess.fork()
    playful.send("Make it punny and animal-themed.")
    serious.send("Make it terse and Unix-flavoured.")

    print("=== playful branch ===")
    print(f"  {playful.history[-1].content}\n")
    print("=== serious branch ===")
    print(f"  {serious.history[-1].content}\n")

    # Rollback: checkpoint before a probe, then restore if we don't like it.
    cid = sess.checkpoint()
    sess.send("Now pitch it as a corporate SaaS product.")
    print("=== probe ===")
    print(f"  {sess.history[-1].content}\n")

    sess.restore(cid)
    print(f"=== after restore — history has {len(sess.history)} messages ===")
    for m in sess.history:
        body = m.content if isinstance(m.content, str) else "[parts]"
        print(f"  [{m.role}] {body[:60]}")
