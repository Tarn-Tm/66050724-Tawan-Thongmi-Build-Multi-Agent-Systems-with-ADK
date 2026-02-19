# 66050724-Tawan-Thongmi-Build-Multi-Agent-Systems-with-ADK
# Workflow
import os
import logging
import google.cloud.logging

from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from google.adk.tools import exit_loop

# Setup Logging
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# --- 1. Tools ---

def save_evidence(tool_context: ToolContext, side: str, evidence: str) -> dict:
    """Saves research data to pos_data or neg_data field in state."""
    try:
        # ตรวจสอบและบันทึกข้อมูลลงใน State
        existing = tool_context.state.get(side, [])
        if isinstance(existing, str): existing = [existing] # ป้องกันกรณีข้อมูลเป็น string
        
        tool_context.state[side] = existing + [evidence]
        logging.info(f"Successfully saved to {side}")
        return {"status": "success"}
    except Exception as e:
        logging.error(f"Error saving evidence to {side}: {e}")
        return {"status": "error", "message": str(e)}

def write_verdict(
    tool_context: ToolContext,
    directory: str,
    filename: str,
    content: str
) -> dict[str, str]:
    """Writes the final court verdict to a .txt file."""
    try:
        # ล้างชื่อไฟล์ให้ไม่มีอักขระพิเศษ
        clean_filename = "".join([c for c in filename if c.isalnum() or c in (' ', '-', '_')]).strip()
        target_path = os.path.join(directory, f"{clean_filename}.txt")
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 2. Leaf Agents ---

admirer_researcher = Agent(
    name="admirer_researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ค้นหาด้านบวกและความสำเร็จ",
    instruction="""
    PROMPT: { PROMPT? }
    เป้าหมาย: ค้นหาข้อมูล "ความสำเร็จและด้านบวก" ของบุคคล/เหตุการณ์ใน PROMPT
    1. ใช้ WikipediaTool ค้นหา (Search) ข้อมูลเชิงบวก
    2. ใช้ tool 'save_evidence' โดยระบุ side='pos_data' และใส่ข้อมูลที่ค้นหาได้ในช่อง evidence
    """,
    tools=[LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())), save_evidence]
)

critic_researcher = Agent(
    name="critic_researcher",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ค้นหาด้านลบและข้อโต้แย้ง",
    instruction="""
    PROMPT: { PROMPT? }
    เป้าหมาย: ค้นหาข้อมูล "ข้อผิดพลาด ข้อโต้แย้ง หรือด้านลบ" ของบุคคล/เหตุการณ์ใน PROMPT
    1. ใช้ WikipediaTool ค้นหา (Search) โดยเน้นคำว่า 'criticism' หรือ 'controversy'
    2. ใช้ tool 'save_evidence' โดยระบุ side='neg_data' และใส่ข้อมูลที่ค้นหาได้ในช่อง evidence
    **สำคัญ: หากหาไม่เจอ ให้ระบุว่าไม่พบข้อมูลที่ขัดแย้งชัดเจนลงใน neg_data**
    """,
    tools=[LangchainTool(tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())), save_evidence]
)

judge = Agent(
    name="judge",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="ตรวจสอบความสมบูรณ์ของข้อมูล",
    instruction="""
    ตรวจสอบข้อมูลปัจจุบัน:
    - POSITIVE DATA: { pos_data? }
    - NEGATIVE DATA: { neg_data? }

    หากข้อมูลทั้งสองฝั่งมีการระบุเนื้อหาแล้ว (แม้ฝั่งใดฝั่งหนึ่งจะแจ้งว่าไม่พบข้อมูล) ให้ใช้ tool 'exit_loop' ทันที
    หากข้อมูลยังว่างเปล่าอยู่ทั้งสองฝั่ง ให้แจ้งทีมสืบสวนหาข้อมูลเพิ่มเติม
    """,
    tools=[exit_loop]
)

verdict_writer = Agent(
    name="verdict_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="สรุปรายงานคำพิพากษา",
    instruction="""
    รวบรวมข้อมูลสรุปสำหรับหัวข้อ: { PROMPT? }
    ข้อมูลด้านบวก: { pos_data? }
    ข้อมูลด้านลบ: { neg_data? }

    1. เขียนรายงานสรุปที่เปรียบเทียบทั้งสองด้านอย่างเป็นกลาง
    2. ใช้ tool 'write_verdict' บันทึกลงไดเรกทอรี 'court_reports' โดยตั้งชื่อไฟล์ตาม { PROMPT? }
    """,
    tools=[write_verdict],
    generate_content_config=types.GenerateContentConfig(temperature=0)
)

# --- 3. Group Agents (Workflow) ---

investigation_team = ParallelAgent(
    name="investigation_team",
    sub_agents=[admirer_researcher, critic_researcher]
)

trial_loop = LoopAgent(
    name="trial_loop",
    sub_agents=[investigation_team, judge],
    max_iterations=2 # ลดจำนวนรอบเพื่อความรวดเร็วและประหยัด quota
)

historical_court_team = SequentialAgent(
    name="historical_court_team",
    sub_agents=[trial_loop, verdict_writer]
)

# --- 4. Root Agent ---

root_agent = Agent(
    name="court_clerk",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    description="เจ้าหน้าที่รับเรื่อง",
    instruction="""
    ทักทายผู้ใช้และรับชื่อบุคคลหรือเหตุการณ์ประวัติศาสตร์
    เมื่อได้รับชื่อแล้ว ให้ใช้ tool 'save_evidence' โดยระบุ side='PROMPT' และเก็บชื่อนั้นไว้
    จากนั้นส่งต่อให้ 'historical_court_team' ทำงาน
    """,
    tools=[save_evidence],
    sub_agents=[historical_court_team]
)

# Parant and subagent
import os
import sys
import logging

sys.path.append("..")
from callback_logging import log_query_to_model, log_model_response
from dotenv import load_dotenv
import google.cloud.logging
from google.adk import Agent
from google.adk.models import Gemini
from google.genai import types
from typing import Optional, List, Dict

from google.adk.tools.tool_context import ToolContext

load_dotenv()

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# --- 1. Tools ---

def save_case_to_state(
    tool_context: ToolContext,
    topic: str
) -> dict[str, str]:
    """Saves the historical topic to state["PROMPT"].

    Args:
        topic (str): The name of the historical figure or event to investigate.

    Returns:
        dict: Status message.
    """
    # บันทึกหัวข้อลงใน State เพื่อส่งต่อให้ทีมสืบสวนใน workflow_agents
    tool_context.state["PROMPT"] = topic
    logging.info(f"[Case Logged] Topic: {topic}")
    return {"status": "success"}

# --- 2. Agents ---

# Agent ช่วยอธิบายขั้นตอน (เหมือนคนให้ข้อมูลที่โต๊ะประชาสัมพันธ์)
court_info_officer = Agent(
    name="court_info_officer",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="อธิบายขั้นตอนการพิจารณาคดีประวัติศาสตร์",
    instruction="""
        - ทักทายผู้ใช้อย่างสุภาพในฐานะเจ้าหน้าที่ประชาสัมพันธ์ของศาลประวัติศาสตร์
        - อธิบายว่าศาลเราจะใช้ AI สองฝั่ง (ฝ่ายชม และ ฝ่ายติ) ไปค้นข้อมูลจาก Wikipedia
        - บอกผู้ใช้ว่าเราจะวนลูปตรวจสอบจนกว่าข้อมูลจะสมดุลที่สุด
        - ถ้าผู้ใช้พิมพ์ 'hello' หรือ 'สวัสดี' ให้ใช้ Agent ตัวนี้ตอบเป็นหลัก
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
)

# Agent คัดกรองหัวข้อ (ตรวจสอบว่าชื่อที่จะสืบสวนโอเคไหม)
case_validator = Agent(
    name="case_validator",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="รับชื่อหัวข้อประวัติศาสตร์และตรวจสอบความพร้อม",
    instruction="""
        - รับชื่อบุคคลหรือเหตุการณ์จากผู้ใช้
        - ตรวจสอบว่าชื่อนั้นชัดเจนพอที่จะค้นหาใน Wikipedia หรือไม่
        - เมื่อได้ชื่อที่แน่นอนแล้ว ให้ใช้ tool 'save_case_to_state' เพื่อบันทึกชื่อนั้นลงในระบบ
        - แจ้งผู้ใช้ว่า "รับเรื่องแล้ว กำลังส่งต่อให้ทีมสืบสวนขนานกัน (Investigation Phase)"
        """,
    before_model_callback=log_query_to_model,
    after_model_callback=log_model_response,
    tools=[save_case_to_state]
)

# Root Agent: ตัวควบคุมทิศทาง (Steering)
root_agent = Agent(
    name="court_reception",
    model=Gemini(model=os.getenv("MODEL"), retry_options=RETRY_OPTIONS),
    description="จุดลงทะเบียนคดีประวัติศาสตร์",
    instruction="""
        - ทักทายผู้ใช้และถามว่าต้องการ 'สอบถามขั้นตอนการทำงาน' หรือ 'ยื่นเรื่องตรวจสอบประวัติศาสตร์'
        - ถ้าผู้ใช้ต้องการทราบข้อมูล หรือทักทายทั่วไป ให้ส่งไปหา 'court_info_officer'
        - ถ้าผู้ใช้บอกชื่อบุคคลหรือเหตุการณ์มาเลย หรือพร้อมจะวิเคราะห์ ให้ส่งไปหา 'case_validator'
        - พยายามทำให้ผู้ใช้รู้สึกเหมือนกำลังคุยกับพนักงานต้อนรับในศาลจริงๆ
        """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.7, # เพิ่มความละมุนในการคุย
    ),
    # รวม Sub Agents ตามโครงสร้าง Parent & Sub-agents
    sub_agents=[court_info_officer, case_validator]
)
