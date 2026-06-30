import pytest
import time
from app.pipeline.live_pipeline import LiveSession, SessionState, InvalidStateTransitionError

class DummyModel:
    def __init__(self, name):
        self.name = name
        self.is_available = True

def test_session_initialization():
    session = LiveSession("test_id", {"dummy": True}, {"dummy": DummyModel("dummy")}, ["dummy"])
    
    # After init, it should be in READY state because __init__ transitions INITIALIZING -> LOADING_MODELS -> READY
    assert session.current_state == SessionState.READY
    assert session.session_uuid is not None
    assert session.worker_uuid is not None
    assert len(session.state_history) == 3
    
    # Check history
    assert session.state_history[0]["state"] == SessionState.INITIALIZING
    assert session.state_history[1]["state"] == SessionState.LOADING_MODELS
    assert session.state_history[2]["state"] == SessionState.READY

def test_valid_transitions():
    session = LiveSession("test_id", {}, {}, [])
    assert session.current_state == SessionState.READY
    
    # Ready -> Streaming
    session.transition_state(SessionState.STREAMING)
    assert session.current_state == SessionState.STREAMING
    
    # Streaming -> Recovering
    session.transition_state(SessionState.RECOVERING)
    assert session.current_state == SessionState.RECOVERING
    
    # Recovering -> Streaming
    session.transition_state(SessionState.STREAMING)
    assert session.current_state == SessionState.STREAMING
    
    # Streaming -> Failed
    session.transition_state(SessionState.FAILED)
    assert session.current_state == SessionState.FAILED
    
    # Failed -> Terminating
    session.transition_state(SessionState.TERMINATING)
    assert session.current_state == SessionState.TERMINATING
    
    # Terminating -> Terminated
    session.transition_state(SessionState.TERMINATED)
    assert session.current_state == SessionState.TERMINATED

def test_invalid_transitions():
    session = LiveSession("test_id", {}, {}, [])
    assert session.current_state == SessionState.READY
    
    # Cannot go Ready -> Initializing
    with pytest.raises(InvalidStateTransitionError):
        session.transition_state(SessionState.INITIALIZING)
        
    session.transition_state(SessionState.FAILED)
    
    # Cannot go Failed -> Streaming
    with pytest.raises(InvalidStateTransitionError):
        session.transition_state(SessionState.STREAMING)

def test_timestamps_and_logging():
    session = LiveSession("test_id", {}, {}, [])
    t0 = session.updated_at
    
    time.sleep(0.01)
    session.transition_state(SessionState.STREAMING, "Test reason")
    
    assert session.updated_at > t0
    assert session.state_history[-1]["reason"] == "Test reason"
    assert session.state_history[-1]["state"] == SessionState.STREAMING
