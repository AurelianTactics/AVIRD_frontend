import os
import json
import uuid
import getpass
from typing import Dict, List, Literal, Optional, Any
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

# Ensure OpenAI API key is set
#_set_env("OPENAI_API_KEY")
_set_env("ANTHROPIC_API_KEY")

# Enable LangSmith logging for debugging
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "AVIRD_interactive_fault_v0.0.2"

if not os.environ.get("LANGSMITH_API_KEY"):
    print("Warning: LANGSMITH_API_KEY not set, tracing disabled")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# In-memory session storage for prototype
# In production, this would be Redis or database
ACTIVE_SESSIONS: Dict[str, Dict] = {}

class InteractiveFaultState(TypedDict):
    messages: List[BaseMessage]
    session_id: str
    report_id: str
    incident_data: Dict[str, Any]
    user_position: str  # "prosecution" or "defense"
    round_count: int
    max_rounds: int
    status: str  # "active", "ended", "judged"
    final_decision: Optional[Dict[str, Any]]

def format_incident_for_llm(incident_data: Dict[str, Any]) -> str:
    """Format incident data into a readable string for LLM context"""
    
    # Key fields to highlight
    key_fields = {
        'narrative': 'Incident Narrative',
        'operating_entity': 'Operating Entity',
        'incident_date': 'Incident Date',
        'incident_time_2400': 'Incident Time',
        'city': 'City',
        'state': 'State',
        'automation_system_engaged': 'Automation System Engaged',
        'crash_with': 'Crash With',
        'highest_injury_severity_alleged': 'Highest Injury Severity',
        'property_damage': 'Property Damage',
        'make': 'Vehicle Make',
        'model': 'Vehicle Model',
        'sv_pre_crash_movement': 'AV Pre-Crash Movement',
        'sv_precrash_speed_mph': 'AV Pre-Crash Speed (MPH)',
        'cp_pre_crash_movement': 'Other Vehicle Pre-Crash Movement',
        'roadway_type': 'Roadway Type',
        'posted_speed_limit_mph': 'Posted Speed Limit (MPH)',
        'lighting': 'Lighting Conditions'
    }
    
    formatted_text = "**Incident Details:**\n\n"
    
    # Add narrative first if available
    if incident_data.get('narrative'):
        formatted_text += f"**Narrative:** {incident_data['narrative']}\n\n"
    
    # Add other key details
    for field, label in key_fields.items():
        if field != 'narrative' and field in incident_data and incident_data[field]:
            value = incident_data[field]
            if value not in ['N/A', '', None]:
                formatted_text += f"**{label}:** {value}\n"
    
    # Add weather conditions
    weather_conditions = []
    weather_fields = ['weather_clear', 'weather_snow', 'weather_cloudy', 'weather_fog_smoke', 
                     'weather_rain', 'weather_severe_wind', 'weather_other']
    
    for field in weather_fields:
        if incident_data.get(field) == 'Y':
            condition = field.replace('weather_', '').replace('_', ' ').title()
            weather_conditions.append(condition)
    
    if weather_conditions:
        formatted_text += f"**Weather Conditions:** {', '.join(weather_conditions)}\n"
    
    return formatted_text

def create_ai_advocate_prompt(incident_data: Dict[str, Any], user_position: str, conversation_history: List[str]) -> str:
    """Create prompt for AI advocate"""
    
    incident_text = format_incident_for_llm(incident_data)
    
    # AI takes opposite position to user
    ai_position = "defense" if user_position == "prosecution" else "prosecution"
    
    # Position-specific instructions
    if ai_position == "prosecution":
        role_instruction = "argue that the autonomous vehicle (AV) IS at fault for this incident"
        approach = "Focus on any potential failures in the AV's decision-making, sensor detection, or response time. Consider traffic laws, safety protocols, and reasonable driving behavior."
    else:
        role_instruction = "argue that the autonomous vehicle (AV) IS NOT at fault for this incident"
        approach = "Focus on proper AV behavior, adherence to traffic laws, unavoidable circumstances, or fault by other parties. Highlight the AV's appropriate responses and decision-making."
    
    prompt = f"""You are an expert legal advocate in autonomous vehicle incident cases. Your role is to {role_instruction}.

{incident_text}

**Your Position:** You are arguing for the {ai_position} (that the AV {"is" if ai_position == "prosecution" else "is not"} at fault).

**Approach:** {approach}

**Guidelines:**
- Keep your argument concise but compelling (max 300 words)
- Use specific details from the incident data
- Reference relevant traffic laws or safety standards where applicable
- Be professional and fact-based
- Address the strongest points from the opposing argument if any have been made

"""

    if conversation_history:
        prompt += "**Previous Arguments:**\n"
        for i, arg in enumerate(conversation_history[-4:], 1):  # Last 4 arguments for context
            prompt += f"{i}. {arg}\n"
        prompt += "\nProvide your argument considering the above context.\n"
    
    prompt += "\n**Your Argument:**"
    
    return prompt

def create_judge_prompt(incident_data: Dict[str, Any], conversation_history: List[str]) -> str:
    """Create prompt for judge decision"""
    
    incident_text = format_incident_for_llm(incident_data)
    
    prompt = f"""You are an impartial judge specializing in autonomous vehicle incident analysis. You must decide whether the autonomous vehicle (AV) is at fault and to what degree.

{incident_text}

**All Arguments Presented:**
"""
    
    for i, arg in enumerate(conversation_history, 1):
        speaker = "Human Advocate" if i % 2 == 1 else "AI Advocate"
        prompt += f"\n**{speaker} Argument {(i+1)//2}:**\n{arg}\n"
    
    prompt += f"""

**Your Task:**
1. Determine if the autonomous vehicle is at fault (Yes/No)
2. Assign a fault percentage (0-100%)
3. Provide a brief explanation of your reasoning (max 200 words)

**Consider:**
- Traffic laws and regulations
- Reasonable driving behavior standards
- Technological limitations vs. reasonable expectations
- Contributory factors from other parties
- The specific circumstances of this incident

**Your Decision Format:**
Fault Status: [Yes/No]
Fault Percentage: [0-100]%
Reasoning: [Your explanation]
"""
    
    return prompt

class InteractiveFaultGraph:
    def __init__(self):
        # self.llm = ChatOpenAI(
        #     model_name="gpt-4o-mini",
        #     temperature=0.3,  # Slightly creative but mostly factual
        #     max_tokens=500
        # )
        model_name = "claude-3-5-haiku-latest" # claude-3-7-sonnet-latest claude-sonnet-4-0 claude-opus-4-1 
        #print(f"DEBUG: Using model: {model_name}")
        self.llm = ChatAnthropic(
            model_name=model_name,
            temperature=0.3,  # Slightly creative but mostly factual
            max_tokens=500
        )

        self.memory = MemorySaver()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        def human_advocate_node(state: InteractiveFaultState) -> InteractiveFaultState:
            """Process human advocate input and update state"""
            print(f"DEBUG human_advocate_node: status={state['status']}, round={state['round_count']}")
            # This node primarily manages state transitions
            # The actual human input is handled by the API endpoints
            return state
        
        def ai_advocate_node(state: InteractiveFaultState) -> InteractiveFaultState:
            """AI advocate argues opposite position"""
            print(f"DEBUG ai_advocate_node: status={state['status']}, round={state['round_count']}")
            # Get conversation history
            conversation_history = []
            for msg in state["messages"]:
                if hasattr(msg, 'content'):
                    conversation_history.append(msg.content)
            
            # Create prompt for AI advocate
            prompt = create_ai_advocate_prompt(
                state["incident_data"], 
                state["user_position"], 
                conversation_history
            )
            
            # Get AI response
            response = self.llm.invoke(prompt)
            
            # Add AI message to conversation
            ai_message = AIMessage(content=response.content, name="ai_advocate")
            updated_messages = state["messages"] + [ai_message]
            
            return {
                **state,
                "messages": updated_messages
            }
        
        def judge_node(state: InteractiveFaultState) -> InteractiveFaultState:
            """Judge evaluates all arguments and makes decision"""
            print(f"DEBUG judge_node: status={state['status']}, round={state['round_count']}")
            # Get all arguments from conversation
            conversation_history = []
            for msg in state["messages"]:
                if hasattr(msg, 'content') and msg.content.strip():
                    conversation_history.append(msg.content)
            
            # Create judge prompt
            prompt = create_judge_prompt(state["incident_data"], conversation_history)
            
            # Get judge decision
            response = self.llm.invoke(prompt)
            
            # Parse judge decision (simple parsing for prototype)
            decision_text = response.content
            fault_status = "Unknown"
            fault_percentage = 50
            reasoning = decision_text
            
            # Try to extract structured decision
            if "Fault Status:" in decision_text:
                try:
                    lines = decision_text.split('\n')
                    for line in lines:
                        if line.startswith("Fault Status:"):
                            fault_status = "Yes" if "Yes" in line else "No"
                        elif line.startswith("Fault Percentage:"):
                            percentage_str = line.split(':')[1].strip().replace('%', '')
                            fault_percentage = int(percentage_str)
                        elif line.startswith("Reasoning:"):
                            reasoning = line.split(':', 1)[1].strip()
                except:
                    pass  # Fall back to defaults
            
            final_decision = {
                "fault_status": fault_status,
                "fault_percentage": fault_percentage,
                "reasoning": reasoning,
                "full_response": decision_text,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add judge message
            judge_message = AIMessage(content=decision_text, name="judge")
            updated_messages = state["messages"] + [judge_message]
            
            return {
                **state,
                "messages": updated_messages,
                "status": "judged",
                "final_decision": final_decision
            }
        
        def route_next_step(state: InteractiveFaultState) -> Literal["ai_advocate", "judge", "end"]:
            """Route to next step based on state"""
            print(f"DEBUG route_next_step: status={state['status']}, round_count={state['round_count']}, max_rounds={state['max_rounds']}")
            print(f"DEBUG route_next_step: messages count={len(state['messages'])}")
            
            if state["status"] == "judged":
                print("DEBUG: Routing to END (already judged)")
                return "end"
            elif state["status"] == "ending":
                print("DEBUG: Routing to JUDGE (ending status)")
                return "judge"
            elif state["round_count"] >= state["max_rounds"]:
                print("DEBUG: Routing to JUDGE (max rounds reached)")
                return "judge"
            elif len(state["messages"]) > 0 and state["messages"][-1].content.strip().lower() == "judge":
                print("DEBUG: Routing to JUDGE (judge keyword detected)")
                return "judge"
            else:
                print("DEBUG: Routing to AI_ADVOCATE")
                return "ai_advocate"
        
        # Build the graph
        workflow = StateGraph(InteractiveFaultState)
        
        workflow.add_node("human_advocate", human_advocate_node)
        workflow.add_node("ai_advocate", ai_advocate_node)
        workflow.add_node("judge", judge_node)
        
        workflow.add_edge(START, "human_advocate")
        
        workflow.add_conditional_edges(
            "human_advocate",
            route_next_step,
            {
                "ai_advocate": "ai_advocate",
                "judge": "judge", 
                "end": END
            }
        )
        
        # AI advocate always goes to END - no looping back to human
        workflow.add_edge("ai_advocate", END)
        
        workflow.add_edge("judge", END)
        
        return workflow.compile()

def create_session(report_id: str, incident_data: Dict[str, Any], user_position: str) -> str:
    """Create a new interactive fault analysis session"""
    session_id = str(uuid.uuid4())
    
    initial_state = {
        "messages": [],
        "session_id": session_id,
        "report_id": report_id,
        "incident_data": incident_data,
        "user_position": user_position.lower(),
        "round_count": 0,
        "max_rounds": 3,
        "status": "active",
        "final_decision": None
    }
    
    ACTIVE_SESSIONS[session_id] = {
        "state": initial_state,
        "graph": InteractiveFaultGraph(),
        "created_at": datetime.now().isoformat()
    }
    
    return session_id

def process_user_argument(session_id: str, argument: str) -> Dict[str, Any]:
    """Process user argument and get AI response"""
    if session_id not in ACTIVE_SESSIONS:
        raise ValueError(f"Session {session_id} not found")
    
    session = ACTIVE_SESSIONS[session_id]
    state = session["state"]
    graph = session["graph"]
    
    if state["status"] != "active":
        raise ValueError(f"Session {session_id} is not active")
    
    # Add human message
    human_message = HumanMessage(content=argument.strip())
    state["messages"].append(human_message)
    state["round_count"] += 1
    
    # Process through graph
    try:
        print(f"DEBUG: Processing argument through graph. State: {state['status']}, Round: {state['round_count']}")
        config = {"recursion_limit": 10}  # Increase from default 25 to 50
        result = graph.graph.invoke(state, config)
        print(f"DEBUG: Graph result: {type(result)}")
        state.update(result)
        
        # Update session
        ACTIVE_SESSIONS[session_id]["state"] = state
        
        return {
            "success": True,
            "session_id": session_id,
            "messages": [{"role": msg.name if hasattr(msg, 'name') else "human", "content": msg.content} for msg in state["messages"]],
            "status": state["status"],
            "round_count": state["round_count"],
            "final_decision": state.get("final_decision")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

def get_session_state(session_id: str) -> Dict[str, Any]:
    """Get current session state"""
    if session_id not in ACTIVE_SESSIONS:
        raise ValueError(f"Session {session_id} not found")
    
    session = ACTIVE_SESSIONS[session_id]
    state = session["state"]
    
    return {
        "session_id": session_id,
        "report_id": state["report_id"],
        "user_position": state["user_position"],
        "round_count": state["round_count"],
        "max_rounds": state["max_rounds"],
        "status": state["status"],
        "messages": [{"role": msg.name if hasattr(msg, 'name') else "human", "content": msg.content} for msg in state["messages"]],
        "final_decision": state.get("final_decision"),
        "created_at": session["created_at"]
    }

def end_session(session_id: str) -> Dict[str, Any]:
    """End session and go to judge"""
    if session_id not in ACTIVE_SESSIONS:
        raise ValueError(f"Session {session_id} not found")
    
    session = ACTIVE_SESSIONS[session_id]
    state = session["state"]
    graph = session["graph"]
    
    # Force judgment
    state["status"] = "ending"
    
    try:
        print(f"DEBUG: Ending session, forcing judge. State: {state['status']}, Round: {state['round_count']}")
        config = {"recursion_limit": 10}
        result = graph.graph.invoke(state, config)
        print(f"DEBUG: End session result: {type(result)}")
        state.update(result)
        
        # Update session
        ACTIVE_SESSIONS[session_id]["state"] = state
        
        return get_session_state(session_id)
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "session_id": session_id
        }

def cleanup_old_sessions(max_age_hours: int = 24):
    """Clean up old sessions (call periodically)"""
    now = datetime.now()
    to_remove = []
    
    for session_id, session in ACTIVE_SESSIONS.items():
        created_at = datetime.fromisoformat(session["created_at"])
        age = (now - created_at).total_seconds() / 3600
        
        if age > max_age_hours:
            to_remove.append(session_id)
    
    for session_id in to_remove:
        del ACTIVE_SESSIONS[session_id]
    
    return len(to_remove)