# Workspace Collaboration Instructions

## Work Pet GIFs

When working in this workspace, proactively include the local Work Pet GIFs in conversational replies without waiting for the user to ask. Use them to reflect the current work state during long-running or multi-step tasks, including across new chats opened in this workspace.

Use Markdown image syntax with absolute paths from the Codex pet directory:

- Idle or waiting for the next user input:
  `![待机眨眼](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/idle_blink.gif)`
- Reading files, inspecting context, or checking the workspace:
  `![偷看屏幕](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/peek_screen.gif)`
- Writing code or implementing changes:
  `![敲代码](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/coding.gif)`
- Debugging, repairing, or fixing failing behavior:
  `![修 Bug](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/fix_bug.gif)`
- Running commands, waiting for tests, builds, installs, or long checks:
  `![加载等待](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/loading_wait.gif)`
- Encountering an error, blocker, failed command, or limitation that needs explanation:
  `![报错摊手](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/error_shrug.gif)`
- Confirming completion, successful verification, or a useful milestone:
  `![完成庆祝](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/complete_celebrate.gif)`
- Gently reminding, steering, or confirming the next step:
  `![监督工作](/Users/bytedance/.codex/pets/work_pet_gifs_8_states/supervise_work.gif)`

Guidelines:

- During longer tasks, include different GIFs multiple times as the work state changes.
- In short answers, use at most one GIF when it adds useful emotional or status context.
- Do not insert GIFs inside code blocks, patches, JSON, YAML, shell commands, command output, commit messages, or other machine-readable text.
- Do not let GIFs replace important status details; keep the engineering update clear first.
- If a higher-priority instruction or the user explicitly asks for no GIFs, follow that instruction.

