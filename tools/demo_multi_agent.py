"""多算法择优演示：把 5 个变化检测工具注册进一个 Agent，让 LLM 选择。

这是「多算法择优」的第一版骨架，包含：
1. 统一注册：5 个算法工具（ChangeFormer/BIT/TinyCD/SNUNet/DDPM-CD）都注册进 Agent。
2. 提示词分析：LLM 读 5 个工具描述，自己挑一个调用。
3. 程序层兜底（反幻觉）：检查 Agent 是否真的调用了工具；没调就警告。

用法：
    set DEEPSEEK_API_KEY=...
    python tools/demo_multi_agent.py --question "..."
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for d in (str(ROOT), str(ROOT / "tools")):
    if d not in sys.path:
        sys.path.insert(0, d)

from lagent.actions import ActionExecutor  # noqa: E402
from lagent.agents import ReAct  # noqa: E402
from lagent.llms import GPTAPI  # noqa: E402

from bit_tool import BITTool  # noqa: E402
from changeformer_tool import ChangeFormerTool  # noqa: E402
from ddpm_cd_tool import DDPMCDTool  # noqa: E402
from snunet_tool import SNUNetTool  # noqa: E402
from tinycd_tool import TinyCDTool  # noqa: E402

# 这 5 个名字 = 各工具的 __tool_name__，用于反幻觉检查
TOOL_NAMES = {"changeformer", "bit", "tinycd", "snunet", "ddpm_cd"}


def build_agent() -> tuple[ReAct, list]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 DEEPSEEK_API_KEY")
    llm = GPTAPI(
        model_type=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        key=api_key,
        openai_api_base=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/chat/completions"),
    )
    tools = [ChangeFormerTool(), BITTool(), TinyCDTool(), SNUNetTool(), DDPMCDTool()]
    return ReAct(llm=llm, action_executor=ActionExecutor(actions=tools)), tools


def main() -> None:
    parser = argparse.ArgumentParser(description="多算法择优演示")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    chatbot, _ = build_agent()
    history = [dict(role="user", content=args.question)]
    result = chatbot.chat(history)

    # 反幻觉检查：哪些工具被真正调用了
    tool_calls = [a for a in result.actions if a.type in TOOL_NAMES]

    print("\n=== 实际调用的工具 ===")
    if tool_calls:
        for a in tool_calls:
            print(f"  工具: {a.type} | 状态: {a.state}")
    else:
        print("  [警告] 未调用任何变化检测工具 —— 可能是 LLM 幻觉（声称有结果却没调工具）")

    print("\n=== Agent 回答 ===")
    print(result.response)


if __name__ == "__main__":
    main()
