# Security Guardrail: Guard `.env`

- Under no circumstances should the agent read, view, parse, grep, or display `.env`.
- Under no circumstances should the agent edit, write, modify, or delete `.env`.
- `.env` is solely maintained by the user.
- Always refer to `.env.example` for key names and templates.
