# AGENTS.md

## Project Mission

Build a competitive Tencent Kaiwu hok1v1 reinforcement-learning agent for the official 1v1 Honor of Kings task. The development boundary is the official Kaiwu code package and protocol in this repository; the practical baseline should be adapted from `wty-yy/kaiwu_taichu/tree/2025/hok_semi` into a version that conforms to our current task.

Primary target:

- Preserve official framework compatibility.
- Use PPO as the algorithm spine.
- Adapt `wty-yy/kaiwu_taichu/tree/2025/hok_semi` into our own task-compliant baseline.
- Develop against `agent_diy` as the project-owned candidate agent that hosts the adapted baseline and later optimizations.
- Keep `agent_ppo` available as the official baseline reference unless a task explicitly targets it.

## Authoritative Context

Read these first when starting a new Codex CLI session:

1. `开悟强化学习系统化调研报告.md`
2. `DEVELOPMENT_PLAN.md`
3. `开发指南/项目简介.md`
4. `开发指南/环境详述.md`
5. `开发指南/数据协议.md`
6. `开发指南/智能体详述.md`
7. `腾讯开悟强化学习框架/综述.md`
8. `腾讯开悟强化学习框架/智能体/特征处理.md`
9. `腾讯开悟强化学习框架/智能体/智能体开发.md`
10. `腾讯开悟强化学习框架/智能体/工作流开发.md`
11. `基线开发参考.md`
12. `champion.md`
13. `reward1.md`
14. `reward2.md`
15. `reward3.md`

Legacy note:

- `调研报告.md` is a historical tactical memo. It is not the main plan, but its high-value content has been migrated into section `5.3` of `开悟强化学习系统化调研报告.md`.

External reference:

- `https://github.com/wty-yy/kaiwu_taichu/tree/2025/hok_semi`

Use the external repository as the high-score baseline source, then re-map it against our official protocol, heroes, action space, feature schema, and platform constraints.

## Repository Map

| Path | Role |
| --- | --- |
| `agent_ppo/` | Official PPO baseline. Use for comparison and selective reference. |
| `agent_diy/` | Project-owned candidate agent. Prefer implementing durable improvements here. |
| `conf/algo_conf_hok1v1.toml` | Agent entrypoint mapping for `ppo` and `diy`. |
| `conf/configure_app.toml` | Training/runtime settings such as batch size and model dump frequency. |
| `kaiwu.json` | Optional opponent model pool IDs for platform evaluation. |
| `train_test.py` | Historical local smoke test. Do not run from Codex; platform/official environment validation is handled outside this agent. |
| `开发指南/` | Official hok1v1 task and protocol docs. |
| `腾讯开悟强化学习框架/` | Official framework docs. |

## Development Rules

- Do not rewrite the framework boundary. Keep official wrappers, data classes, workflow signatures, model save/load contracts, and action protocol compatible.
- It is acceptable to borrow and rewrite `kaiwu_taichu/hok_semi` code, but the resulting code must conform to this repository's official task protocol.
- Do not delete `agent_ppo` baseline files as part of optimization work.
- Implement the adapted high-score baseline in `agent_diy`; keep `agent_ppo` as the Tencent official baseline reference.
- Keep feature dimensions, `Config.DATA_SPLIT_SHAPE`, `Config.SAMPLE_DIM`, model input shapes, and sample serialization in lockstep.
- Any action head change must be audited with `LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]`, `legal_action`, and `sub_action_mask`.
- Any reward change must expose per-item monitoring or loggable sub-values.
- Any self-play change must preserve `common_ai`, `selfplay`, and custom model ID paths.
- Avoid speculative algorithm swaps. PPO remains the default unless there is a validated reason.
- The agent must handle all four 112/133 matchups: `112 vs 112`, `112 vs 133`, `133 vs 112`, and `133 vs 133`. Do not optimize only for cross-hero play.

## Module Ownership

Use these roles to split work and reviews:

| Role | Owned Scope | Main Files |
| --- | --- | --- |
| Feature Agent | Observation parsing, feature normalization, mirror transforms, candidate target features | `agent_diy/feature/`, later selected `agent_ppo/feature/` references |
| Reward Agent | Reward items, zero-sum differencing, scenario rewards, reward monitoring | `agent_diy/feature/reward_process.py`, config reward dicts |
| Model Agent | Encoders, recurrent branch, target attention, action heads, value head | `agent_diy/model/model.py` |
| Algorithm Agent | PPO loss, dual-clip, advantage normalization, value clipping, gradient stability | `agent_diy/algorithm/algorithm.py`, model loss helpers |
| Workflow Agent | common_ai warmup, self-play, opponent pool, PFSP, model loading, eval config | `agent_diy/workflow/train_workflow.py`, `kaiwu.json` |
| Evaluation Agent | Metrics, static/shape checks, regression checks, candidate model selection | monitor builders, logs, lightweight local probes |

## Hok Semi Adaptation Policy

When adapting from `kaiwu_taichu/hok_semi`:

1. Start from the `hok_semi` implementation shape: structured features, reward manager, PPO loss, recurrent/MLP fusion, target attention, and monitoring.
2. Replace semi-final assumptions with our current task assumptions: heroes `112/133`, current official docs, action target ordering, and local workflow constraints.
3. Put the adapted implementation in `agent_diy`.
4. Keep `agent_ppo` unchanged as a comparison baseline.
5. Add shape assertions and monitoring at every protocol boundary.
6. Run static/shape checks instead of `python train_test.py`.
7. Record the static/shape check result in the development plan checklist.

Hero adaptation requirement:

- Use shared common hero features plus hero-specific feature branches or masks.
- Include hero ID conditioning in the policy input.
- Treat same-hero matchups as first-class cases, not edge cases.

Priority ideas to absorb:

- Structured unit features.
- Position and unit encoders.
- Reward manager structure.
- Target attention.
- MLP plus recurrent fusion.
- Value normalization only after the base PPO path is stable.

## Validation

Minimum checks after code changes:

- Do not run `python train_test.py` under any circumstance in Codex.
- Use static/shape checks instead, such as `python -m py_compile ...`, `Config.validate()`, targeted reward/action helper checks, and model forward/sample-shape probes when relevant.
- Platform or official Kaiwu runtime validation is handled outside Codex.

Additional checks when relevant:

- Inspect `Config.SAMPLE_DIM` against generated sample arrays.
- Verify `predict` and `exploit` both apply legal action masking.
- Verify red/blue mirror transforms for every location-like field.
- Verify `agent.init_config()` does not randomly choose summoner skills in final candidate mode.
- Verify reward item names and weights match config and monitor output.

## First Principles

- Win condition is tower destruction, not reward maximization by itself.
- Training reward is diagnostic, not proof of strength.
- Simple stable PPO with correct features beats complex unstable experiments.
- Every feature, reward, and loss term must have a reason tied to observable gameplay behavior.
