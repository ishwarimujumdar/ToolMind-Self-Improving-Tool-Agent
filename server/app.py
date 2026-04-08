import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from openenv.core.env_server import create_fastapi_app
from server.environment import ToolCallEnv
from models import ToolCallAction, ToolCallObservation


def main():
    return create_fastapi_app(
        ToolCallEnv,
        action_cls=ToolCallAction,
        observation_cls=ToolCallObservation,
    )


app = main()


if __name__ == "__main__":
    main()
