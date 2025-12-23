"""Spawn the build_engine function asynchronously."""
import modal
from build_engine import app, build_engine

if __name__ == "__main__":
    with app.run():
        # Spawn the build function (returns immediately)
        call = build_engine.spawn()
        print(f"Build spawned with call ID: {call.object_id}")
        print("The build will continue on Modal servers.")
        print("Check the Modal dashboard to monitor progress.")
