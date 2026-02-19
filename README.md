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
