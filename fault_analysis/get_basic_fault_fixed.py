'''
Use Langraph to get the incident, fault, fault %, and short explanation

Usage
python fault_analysis/get_basic_fault_fixed.py --fault_version mvp_0.01 --target local

For DB insert to work, table must be created first in database_management/create_fault_table.py
'''

import getpass
import os
import traceback
import json
import sys
from sqlalchemy import create_engine, text

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("python-dotenv not installed, relying on system environment variables")

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAI, ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from typing import Annotated

import argparse
import copy 
import time
import numpy as np
import pandas as pd
from datetime import datetime


def run_get_fault(argparse_args):
    """
    Run the fault getting process
    """
    # setup
    _set_env("OPENAI_API_KEY")
    _set_env("LANGSMITH_API_KEY")
    _set_env("DATABASE_PUBLIC_URL")

    # Disable LangSmith tracing for now to avoid API errors
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    # os.environ["LANGCHAIN_PROJECT"] = "AVIRD_fault_v0.0.2"

    # load and treat data
    sgo_df = load_treat_data(argparse_args)

    # make graph
    fault_graph = make_fault_graph(argparse_args, sgo_df)

    # run through each incident
    fault_results_dict = {}
    config = {"recursion_limit": 10}

    for i in range(len(sgo_df)):
        print(f"Processing incident {i}")
        # run through graph
        incident_message = get_incident_message(sgo_df, i)
        
        events = fault_graph.stream(
            {
                "messages": [
                    HumanMessage(content=incident_message),
                ],
            },
            config,
        )

        for event in events:
            #print(event)
            fault_results_dict = add_to_results_dict(fault_results_dict, event, sgo_df, i)

    # save results
    save_treat_results(argparse_args, fault_results_dict)


def add_to_results_dict(fault_results_dict, event, sgo_df, i):
    '''
    to do:
    better treatment of if sgo id not found/ error
    '''
    
    sgo_id = sgo_df['report_id'].iloc[i]

    try:
            
        if sgo_id not in fault_results_dict:
        
                fault_results_dict[sgo_id]= {}
                # Clean the message content before parsing
                event_content = clean_ai_message(event['arbitrator']['messages'][0].content)
                parsed_event_content = json.loads(event_content)

                if "is_av_at_fault" in parsed_event_content and isinstance(parsed_event_content["is_av_at_fault"], bool):
                    fault_results_dict[sgo_id]["is_av_at_fault"] = parsed_event_content["is_av_at_fault"]
                else:
                    fault_results_dict[sgo_id]["is_av_at_fault"] = None
                    print(f"Warning: is_av_at_fault not in results or not a bool {parsed_event_content}")

                if "av_fault_percentage" in parsed_event_content and isinstance(parsed_event_content["av_fault_percentage"], (float, int)) \
                    and 0 <= parsed_event_content["av_fault_percentage"] <= 1:
                    fault_results_dict[sgo_id]["av_fault_percentage"] = parsed_event_content["av_fault_percentage"] 
                else:
                    fault_results_dict[sgo_id]["av_fault_percentage"] = None
                    print(f"Warning: av_fault_percentage not in results or not a float between 0 and 1 {parsed_event_content}")

                if "short_explanation_of_decision" in parsed_event_content and isinstance(parsed_event_content["short_explanation_of_decision"], str):
                    fault_results_dict[sgo_id]["short_explanation_of_decision"] = parsed_event_content["short_explanation_of_decision"][:1000]
                else:
                    fault_results_dict[sgo_id]["short_explanation_of_decision"] = None
                    print(f"Warning: short_explanation_of_decision not in results or not a string {parsed_event_content}")
        else:
            print(f"Warning: sgo_id already exists in dictionary {sgo_id}")

    except Exception as e:
        print("Error: unable to get event message into needed format not found ", str(e))
        trc = traceback.format_exc()
        print(trc)

        fault_results_dict[sgo_id] = {
            "is_av_at_fault": None,
            "av_fault_percentage": None,
            "short_explanation_of_decision": "Error in parse"
        }

    return fault_results_dict


def clean_ai_message(content: str) -> str:
    """Clean the AI message content by removing markdown code blocks."""
    # Remove ```json and ``` markers
    content = content.replace('```json\n', '').replace('\n```', '')
    return content


def get_incident_message(sgo_df, i):
    # Map column names from current database to original CSV names
    check_cols = [
        'lighting', 'roadway_type', 'roadway_surface', 'roadway_description', 'posted_speed_limit_mph',
        'crash_with', 'weather__clear',
        'weather__snow',
        'weather__cloudy', 
        'weather__fog_smoke',
        'weather__rain',
        'weather__severe_wind',
        'weather__unknown',
        'weather__other',
        'weather__other_text',
        'crash_with',

        'cp_pre_crash_movement',
        'cp_any_air_bags_deployed',
        'cp_was_vehicle_towed',
        'cp_contact_area__rear_left',
        'cp_contact_area__left',
        'cp_contact_area__front_left',
        'cp_contact_area__rear',
        'cp_contact_area__top',
        'cp_contact_area__front',
        'cp_contact_area__rear_right',
        'cp_contact_area__right',
        'cp_contact_area__front_right',
        'cp_contact_area__bottom',
        'cp_contact_area__unknown',
        'sv_pre_crash_movement',
        'sv_any_air_bags_deployed',
        'sv_was_vehicle_towed',
        'sv_precrash_speed_mph',
        'sv_pre_crash_speed__unknown',
        'sv_contact_area__rear_left',
        'sv_contact_area__left',
        'sv_contact_area__front_left',
        'sv_contact_area__rear',
        'sv_contact_area__top',
        'sv_contact_area__front',
        'sv_contact_area__rear_right',
        'sv_contact_area__right',
        'sv_contact_area__front_right',
        'sv_contact_area__bottom',
        'sv_contact_area__unknown',
        'city',
        'state',
        'incident_time_2400',
        'make',
        'model',]

    # Get available columns (handle missing columns gracefully)
    available_cols = [col for col in check_cols if col in sgo_df.columns]
    subset_df = sgo_df[available_cols].iloc[[i]].copy()

    # Create human-readable column names
    column_mapping = {
        'make': 'Autonomous Vehicle Make',
        'model': 'Autonomous Vehicle Model',
        'cp_pre_crash_movement': 'Crash Partner Pre-Crash Movement',
        'cp_any_air_bags_deployed': 'Crash Partner Any Air Bags Deployed?',
        'cp_was_vehicle_towed': 'Crash Partner Was Vehicle Towed?',
        'sv_pre_crash_movement': 'Autonomous Vehicle Pre-Crash Movement',
        'sv_any_air_bags_deployed': 'Autonomous Vehicle Any Air Bags Deployed?',
        'sv_was_vehicle_towed': 'Autonomous Vehicle Was Vehicle Towed?',
        'sv_precrash_speed_mph': 'Autonomous Vehicle Precrash Speed (MPH)',
        'incident_time_2400': 'Incident Time (24:00)',
        'posted_speed_limit_mph': 'Posted Speed Limit (MPH)',
    }
    
    subset_df.rename(columns=column_mapping, inplace=True)

    narrative = sgo_df['narrative'].iloc[i] if 'narrative' in sgo_df.columns else "No narrative available"

    incident_message = f'''
        Incident Information:
        - Narrative:
        {narrative}

        - Other Information:
    '''

    for k, v in subset_df.iloc[0].to_dict().items():
        if v is None or pd.isnull(v) or (isinstance(v, str) and v.strip() == '') or \
            (isinstance(v, float) and np.isnan(v)):
            continue

        if v == 'Y':
            treated_v = 'Yes'
        elif v == 'N':
            treated_v = 'No'
        else:
            treated_v = copy.deepcopy(v)
        incident_message += f'{k}: {treated_v}\n'
    
    return incident_message


class State(TypedDict):
    messages: Annotated[list, add_messages]


def make_fault_graph(argparse_args, sgo_df): 
    # Create the chain for arbitrating
    system_prompt = """
    You are a neutral insurance adjuster. Your job is to determine who is at fault for the incident.
    You will provide your answer in JSON format with the following keys: 
    "is_av_at_fault: bool"
    "av_fault_percentage: float in [0, 1]"
    "short_explanation_of_decision: string"
    """

    fault_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    llm = ChatOpenAI(
        model_name="gpt-4o-mini"
        #model_name="gpt-4o"
        #model_name="gpt-3.5-turbo"
        #o1, o1 mini
        # temperature, max tokens
        )

    arbitrator_chain = fault_prompt | llm

    def arbitrator_node(state: State) -> State:
        # Use invoke instead of ainvoke for synchronous operation
        return {"messages": [arbitrator_chain.invoke(state["messages"])]}

    workflow = StateGraph(State)

    # Define the nodes
    workflow.add_node("arbitrator", arbitrator_node)

    # Add the edges
    workflow.set_entry_point("arbitrator")
    workflow.set_finish_point("arbitrator")

    # Compile
    return workflow.compile()


def save_treat_results(argparse_args, fault_results_dict):
    # Generate fault version
    fault_version = get_fault_version(argparse_args)
    
    # Save to file
    save_to_file(argparse_args, fault_results_dict)
    
    # Save to database
    save_to_database(argparse_args, fault_results_dict, fault_version)


def get_fault_version(argparse_args):
    """Get the fault version from args or generate a default one."""
    if hasattr(argparse_args, 'fault_version') and argparse_args.fault_version:
        return argparse_args.fault_version
    return "default_v1"


def save_to_file(argparse_args, fault_results_dict):
    # Create save directory if it doesn't exist
    os.makedirs(argparse_args.save_dir, exist_ok=True)
    
    # Save to JSON file
    save_path = os.path.join(argparse_args.save_dir, 'fault_analysis_results.json')
    with open(save_path, 'w') as f:
        json.dump(fault_results_dict, f, indent=4)
    print(f"Results saved to: {save_path}")


def save_to_database(argparse_args, fault_results_dict, fault_version):
    """Insert fault analysis results into the database."""
    try:
        # Get database connection based on target
        if argparse_args.target == 'railway':
            # For Railway deployment
            db_url = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')
            if not db_url:
                raise ValueError("DATABASE_PUBLIC_URL or DATABASE_URL not found for Railway target")
        else:
            # For local development - use SQLite
            db_url = "sqlite:///./avird_data.db"
        
        engine = create_engine(db_url)
        print(f"Connecting to database: {db_url}")

        # Prepare the data for insertion
        records = []
        for report_id, data in fault_results_dict.items():
            records.append({
                'report_id': report_id,
                'fault_version': fault_version,
                'is_av_at_fault': data.get('is_av_at_fault'),
                'av_fault_percentage': data.get('av_fault_percentage'),
                'short_explanation_of_decision': data.get('short_explanation_of_decision'),
            })

        # Insert records
        with engine.connect() as conn:
            for record in records:
                query = """
                    INSERT INTO fault_analysis (
                        report_id,
                        fault_version,
                        is_av_at_fault, 
                        av_fault_percentage, 
                        short_explanation_of_decision
                    ) 
                    VALUES (
                        :report_id,
                        :fault_version,
                        :is_av_at_fault, 
                        :av_fault_percentage, 
                        :short_explanation_of_decision
                    );
                """
                conn.execute(text(query), record)
                conn.commit()
        
        print(f"Successfully saved {len(records)} records to database with version {fault_version}")
        
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise


def load_treat_data(argparse_args):
    # Use database instead of CSV for current data
    if argparse_args.target == 'railway':
        db_url = os.getenv('DATABASE_PUBLIC_URL') or os.getenv('DATABASE_URL')
    else:
        db_url = "sqlite:///./avird_data.db"
    
    engine = create_engine(db_url)
    
    # Load data from database
    with engine.connect() as conn:
        # Get all incidents where ADS was engaged
        query = text("SELECT * FROM incident_reports WHERE automation_system_engaged = 'ADS'")
        sgo_df = pd.read_sql(query, conn)
    
    print(f"Loaded {len(sgo_df)} ADS-engaged incidents from database")
    return sgo_df


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default='data/fault_analysis', 
                       help="Directory to save fault analysis results. Will be created if it doesn't exist.")
    parser.add_argument("--fault_version", type=str, required=True,
                       help="Version identifier for this fault analysis run.")
    parser.add_argument("--target", type=str, choices=['local', 'railway'], default='local',
                       help="Target database: 'local' for SQLite, 'railway' for PostgreSQL")
    args = parser.parse_args()

    run_get_fault(args)