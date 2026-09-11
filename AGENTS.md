# Agent Directives & Security Guardrails

## STRICT GUARD: Never Read or Edit `.env`
- **CRITICAL CONSTRAINT**: The assistant must **NEVER** read, view, open, inspect, grep, or display the contents of `.env` under any circumstances (including using `view_file`, `grep_search`, shell commands like `cat`, `type`, `Get-Content`, or script executions).
- **CRITICAL CONSTRAINT**: The assistant must **NEVER** edit, write to, modify, overwrite, append to, or delete `.env`.
- The user will manage and input secrets into `.env` manually.
- When referencing configuration or environment variable names, consult `.env.example` only.
- If a new environment variable is introduced, document it in `.env.example` and ask the user to add it to their `.env`.
