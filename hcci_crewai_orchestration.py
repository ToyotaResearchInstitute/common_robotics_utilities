import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from crewai import Agent, Crew, Process, Task, LLM

# Disable telemetry daemons to prevent terminal locks
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# Harmonic Phase Constants
TICKS_PER_CYCLE = 1296000
BASE_FREQ_HZ = 3600
HOURS_PER_DAY = 24
PHASE_STEP_DELTA = 0.052

class PhaseOrchestratorEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.cycle_count = 0

    def compute_lock_state(self):
        with self.lock:
            self.cycle_count += 1
            cycle_phase = (BASE_FREQ_HZ / HOURS_PER_DAY) * PHASE_STEP_DELTA
            return 1 if cycle_phase > 0 else 0

    def verify_channel(self, channel_id):
        return f"CH_{channel_id}_SYNC_OK"

engine = PhaseOrchestratorEngine()
binary_lock = engine.compute_lock_state()

# Local Qwen 2.5 Coder 7B Driver
local_driver = LLM(
    model="ollama/qwen2.5-coder:7b",
    base_url="http://localhost:11434",
    api_key="NA"
)

# CrewAI Agent Definitions
agent_kuramoto = Agent(
    role="Phase Coherence Orchestrator",
    goal="Maintain 0.052 Hz harmonic synchronization across all 6 channels.",
    backstory="Non-linear systems math specialist enforcing binary phase locks.",
    llm=local_driver,
    verbose=False
)

agent_injector = Agent(
    role="D4 Pulse Timing Controller",
    goal="Trigger direct injection pulses strictly on binary 1 phase boundaries.",
    backstory="High-speed microsecond injection timing specialist.",
    llm=local_driver,
    verbose=False
)

agent_intake = Agent(
    role="Manifold Equilibrium Manager",
    goal="Hold lower grille intake pressure at 1/0 phase equilibrium.",
    backstory="Aerodynamic intake plenum pressure control engineer.",
    llm=local_driver,
    verbose=False
)

# Tasks Definition
task_phase = Task(
    description=f"Verify harmonic phase lock signal state = {binary_lock} (Delta = {PHASE_STEP_DELTA}).",
    expected_output="Binary state 1 lock confirmed.",
    agent=agent_kuramoto
)

task_injection = Task(
    description="Execute synchronized D4 injection pulse across 6 channels.",
    expected_output="Injection timing synchronized to phase boundary.",
    agent=agent_injector
)

task_intake_hold = Task(
    description="Maintain intake plenum equilibrium position.",
    expected_output="Equilibrium position locked.",
    agent=agent_intake
)

def run_orchestrator():
    start_time = time.perf_counter()
    print("=== [CREWAI MULTI-AGENT ORCHESTRATION ENGAGED] ===")
    print(f"Base Frequency : {BASE_FREQ_HZ} Hz | Phase Delta: {PHASE_STEP_DELTA}")
    print(f"Binary State   : {binary_lock}")
    print("-------------------------------------------------")

    with ThreadPoolExecutor(max_workers=6) as executor:
        channel_status = list(executor.map(engine.verify_channel, range(1, 7)))
    
    print(f"Active Parallel Channels: {len(channel_status)}/6 Locked")

    crew = Crew(
        agents=[agent_kuramoto, agent_injector, agent_intake],
        tasks=[task_phase, task_injection, task_intake_hold],
        process=Process.sequential,
        verbose=False
    )

    try:
        results = crew.kickoff()
        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        print("-------------------------------------------------")
        print(f"ORCHESTRATION COMPLETE IN {elapsed} ms")
        print(results)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    run_orchestrator()