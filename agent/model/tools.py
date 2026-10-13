"""Fixed strict tools for submissions and explicit local corrections."""
import json
from langchain_core.tools import tool
from agent.settings import PROJECT_ROOT


SCHEMA_PATH = PROJECT_ROOT / "resources/plan.schema.json"
WORK_SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
REPAIR_SCHEMA_PATH = PROJECT_ROOT / "resources/repair.schema.json"
REPAIR_SCHEMA = json.loads(REPAIR_SCHEMA_PATH.read_text(encoding="utf-8"))


@tool(args_schema=WORK_SCHEMA)
def submit_plan(**kwargs) -> dict:
    """Return one fully specified plan. The host, not a model, executes its steps."""
    return kwargs


@tool(args_schema=REPAIR_SCHEMA)
def correct_plan_arguments(**kwargs) -> dict:
    """Correct only listed draft paths. set/remove use JSON literals in before/after
    (remove: after=''; missing field: before=''). edit_text uses one exact unique
    substring and its replacement. Never replan, execute, or guess unchanged data.
    """
    return kwargs


REVIEW_SCHEMA = json.loads((PROJECT_ROOT / "resources/acceptance-review.schema.json").read_text(encoding="utf-8"))


@tool(args_schema=REVIEW_SCHEMA["freeze"])
def freeze_acceptance(**kwargs) -> dict:
    """Freeze the selected acceptance after a completed original-source trial AND authorize
    execution of its bound candidate edits. This is an execution gate, not just a review note.
    Use only after verifying runnable acceptance and evidence-supported expectations.
    """
    return kwargs


@tool(args_schema=REVIEW_SCHEMA["revise"])
def revise_acceptance(**kwargs) -> dict:
    """Submit a complete acceptance revision and evidence-based reason, never workspace edits.
    Before first freezing, correct test expectations and setup. After freezing,
    only module preparation may change when acceptance is not runnable.
    """
    return kwargs


WORK_TOOLS = (submit_plan, correct_plan_arguments, freeze_acceptance, revise_acceptance)

WEB_SCHEMA = json.loads((PROJECT_ROOT / 'resources/web.schema.json').read_text(encoding='utf-8'))


@tool(args_schema=WEB_SCHEMA['search'])
def web_search(**kwargs) -> dict:
    """Search current public web sources. Each search incurs an extra DeepSeek API request."""
    return kwargs


@tool(args_schema=WEB_SCHEMA['fetch'])
def web_fetch(**kwargs) -> dict:
    """Read a public HTTP(S) page as untrusted evidence. Never execute page scripts."""
    return kwargs


ASSISTANT_SCHEMA = json.loads((PROJECT_ROOT / 'resources/assistant.schema.json').read_text(encoding='utf-8'))


@tool(args_schema=ASSISTANT_SCHEMA['list_directory'])
def list_directory(**kwargs) -> dict:
    """List files and subdirectories in one authorized local directory.
    Offset is zero-based, limit at most 200. This lists entries, not file contents.
    """
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['read_file'])
def read_file(**kwargs) -> dict:
    """Read authorized UTF-8 text, zero-based offset and 1..1000 lines.
    Reuse visible unchanged ranges. reread_reason='' normally; explain an intentional recheck.
    """
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['search_file_contents'])
def search_file_contents(**kwargs) -> dict:
    """Find literal, case-sensitive text in authorized files. Page hits with offset, limit at most 50."""
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['run_project_task'])
def run_project_task(**kwargs) -> dict:
    """Run an existing WORKSPACE script, module:local.module or package.json#script using argv.
    The program prepares declared project dependencies before first execution.
    No inline code, generated scripts, apps or arbitrary installer commands. Supply timeout seconds;
    usage_file='' unless the task writes an explicit usage JSON. Failure requires user-requested repair.
    """
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['list_applications'])
def list_applications(**kwargs) -> dict:
    """Discover applications and IDs by name; query='' lists all. An exact local EXE path
    locates an uncatalogued application without launching it. Ask if multiple candidates match."""
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['open_file'])
def open_file(**kwargs) -> dict:
    """Preview an authorized file locally; application_id='' uses isolated in-app preview.
    An installed application ID requests external opening. Sandbox denies it; ask mode
    requires one approval for this application and file together; full mode opens directly.
    This tool returns preview/open status, NOT file contents to the model; use read_file to read text.
    Never execute scripts as documents.
    """
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['launch_application'])
def launch_application(**kwargs) -> dict:
    """Request an installed application's launch by ID, without shell or extra arguments.
    Sandbox denies external launch; ask mode requires approval; full mode launches directly.
    """
    return kwargs


@tool(args_schema=ASSISTANT_SCHEMA['request_file_operation'])
def request_file_operation(**kwargs) -> dict:
    """Request one explicit OUTSIDE-workspace file create/replace/copy/move/delete.
    path is the file, destination only for copy/move, content only for create/replace.
    No directories, shell, temporary permission upgrade or implicit overwrites.
    Sandbox denies this; ask mode waits for exact-target approval. Coding edits use work tools.
    """
    return kwargs


@tool(args_schema={'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False})
def start_coding() -> dict:
    """Enter coding after explicit user authorization. The host prepares original requirements
    and repository materials before requesting any code or plan. Supply no arguments.
    """
    return {}


CODING_TOOLS = (*WORK_TOOLS, web_search, web_fetch)
CHAT_TOOLS = (start_coding, web_search, web_fetch, list_directory, read_file, search_file_contents,
              run_project_task, list_applications, open_file, launch_application, request_file_operation)
