from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import ToolCallAction, ToolCallObservation, ToolCallState, Scenario, ToolDefinition


class ToolCallEnvClient(EnvClient[ToolCallAction, ToolCallObservation, ToolCallState]):
    """Client for interacting with the Tool Call RL Environment."""

    def _step_payload(self, action: ToolCallAction) -> dict:
        """Convert action to JSON payload."""
        return {
            "scenario_id": action.scenario_id,
            "tool_calls": action.tool_calls,
            "should_refuse": action.should_refuse,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse server JSON response into StepResult."""
        tool_defs = [ToolDefinition(**td) for td in payload.get("tool_definitions", [])]
        obs = ToolCallObservation(
            scenario=Scenario(**payload["scenario"]),
            tool_definitions=tool_defs,
            queue_size=payload["queue_size"],
            current_step=payload["current_step"],
            reward=payload["reward"],
            done=payload["done"],
        )
        return StepResult(
            observation=obs,
            reward=payload["reward"],
            done=payload["done"],
        )

    def _parse_state(self, payload: dict) -> ToolCallState:
        """Parse server JSON response into ToolCallState."""
        return ToolCallState(
            current_index=payload["current_index"],
            total_scenarios=payload["total_scenarios"],
            processed_scenario_ids=payload["processed_scenario_ids"],
            score=payload["score"],
            done=payload["done"],
        )
