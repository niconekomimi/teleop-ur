"""GUI-side intent controller for orchestrator-backed action gating."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import ControlCoordinator, SystemOrchestrator, TransitionDecision


@dataclass(frozen=True)
class IntentResult:
    allowed: bool
    reason: str = ""
    title: str = ""
    message: str = ""


class GuiIntentController:
    _REASON_MESSAGES = {
        "inference_executing": "推理任务正在直接控制机器人，请先停止任务执行。",
        "teleop_running": "遥操作系统正在输出控制命令，请先停止遥操作系统。",
        "inference_not_ready": "请先启动推理并等待模型就绪，再开始执行任务。",
        "estopped": "系统当前处于急停状态，请先停止当前推理流程后再重新进入控制。",
        "homing_active": "机器人当前正在执行回 Home，请等待当前动作完成。",
        "home_zone_active": "机器人当前正在执行 Home Zone，请等待当前动作完成。",
    }

    def __init__(self, orchestrator: SystemOrchestrator | ControlCoordinator) -> None:
        self._orchestrator = orchestrator

    def _result_from_decision(self, decision: TransitionDecision, *, title: str = "控制冲突") -> IntentResult:
        if decision.allowed:
            return IntentResult(True)
        message = self._REASON_MESSAGES.get(decision.reason, f"当前请求被状态机拒绝: {decision.reason}")
        return IntentResult(False, reason=decision.reason, title=title, message=message)

    def check_start_teleop(self) -> IntentResult:
        return self._result_from_decision(self._orchestrator.request_start_teleop())

    def check_enable_inference_execution(self) -> IntentResult:
        return self._result_from_decision(self._orchestrator.request_enable_inference_execution())

    def check_recording(self, active: bool) -> IntentResult:
        decision = (
            self._orchestrator.request_start_recording()
            if active
            else self._orchestrator.request_stop_recording()
        )
        return self._result_from_decision(decision, title="录制请求被拒绝")

    def check_commander_motion(self, motion: str) -> IntentResult:
        if motion == "home":
            decision = self._orchestrator.request_go_home()
        else:
            decision = self._orchestrator.request_go_home_zone()
        return self._result_from_decision(decision, title="控制冲突")
