以下のプログラムを、streamlitを使わず、.tsxなどのより本番に特化したプログラムで書いてください。
docker build "salesgenius" .
でビルドするためのdockerfileを作成してください。

import os
import sys
import json
from pathlib import Path
import base64
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
# import faiss  # type: ignore # FAISSは使わない
import numpy as np
from openai import AzureOpenAI
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions

# ---------------------------------------------------------------
# 1. 環境変数ロード
# ---------------------------------------------------------------
load_dotenv()
REQUIRED_VARS = [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_KEY",
    "AZURE_OPENAI_CHAT_DEPLOYMENT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_SEARCH_ENDPOINT",
    "AZURE_SEARCH_KEY",
    "AZURE_SEARCH_INDEX",
    "AZURE_BLOB_CONN_STR",
    "AZURE_BLOB_CONTAINER",
]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    sys.exit(f"[FATAL] .env is missing: {', '.join(missing)}")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
CHAT_DEPLOYMENT = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
EMBED_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# Azure AI Search
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX")

# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_BLOB_CONN_STR")
AZURE_STORAGE_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER")

# ---------------------------------------------------------------
# 2. デバッグ用ヘルパー
# ---------------------------------------------------------------
DEBUG = os.getenv("DEBUG", "1") != "0"
def dbg(*args, **kwargs):
    if DEBUG:
        print("[DBG]", *args, **kwargs)

# ---------------------------------------------------------------
# 3. OpenAI クライアント初期化
# ---------------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)
dbg("Azure endpoint:", AZURE_OPENAI_ENDPOINT)

# ---------------------------------------------------------------
# 4. Azure Blob Storage & AI Search クラス
# ---------------------------------------------------------------
class AzureBlobStorageManager:
    def __init__(self, connection_string: str, container_name: str):
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)

    def list_files(self, prefix="") -> list[str]:
        """指定されたプレフィックスを持つファイルの一覧を返す"""
        return [blob.name for blob in self.container_client.list_blobs(name_starts_with=prefix)]

    def read_file(self, blob_name: str) -> str:
        """指定されたファイルの内容を文字列として返す"""
        blob_client = self.container_client.get_blob_client(blob_name)
        return blob_client.download_blob().readall().decode("utf-8")

    def get_sas_url(self, blob_name: str) -> str:
        """ファイルのダウンロード用SAS URLを生成する"""
        sas_token = generate_blob_sas(
            account_name=self.container_client.account_name,
            container_name=self.container_client.container_name,
            blob_name=blob_name,
            account_key=self.blob_service_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1) # 1時間有効
        )
        return f"https://{self.container_client.account_name}.blob.core.windows.net/{self.container_client.container_name}/{blob_name}?{sas_token}"

class AzureAISearcher:
    def __init__(self, endpoint: str, key: str, index_name: str):
        self.search_client = SearchClient(endpoint, index_name, AzureKeyCredential(key))

    def _embed(self, text: str) -> list[float]:
        res = client.embeddings.create(input=text, model=EMBED_DEPLOYMENT)
        return res.data[0].embedding

    def search(self, query: str, filter_expression: str | None = None, top_k: int = 5) -> list[dict]:
        vector_query = VectorizedQuery(
            vector=self._embed(query),
            k_nearest_neighbors=top_k,
            fields="vector"
        )

        results = self.search_client.search(
            search_text=None, # テキスト検索は使わない
            vector_queries=[vector_query],
            filter=filter_expression,
            select=["id", "path", "json_path", "md_path", "original_path"],
        )

        search_results = []
        for r in results:
            search_results.append({
                'id': r['id'],
                'path': r['path'],
                'json_path': r['json_path'],
                'md_path': r['md_path'],
                'original_path': r['original_path'],
                'score': r['@search.score']
            })
        return search_results

# クライアントの初期化
blob_manager = AzureBlobStorageManager(AZURE_STORAGE_CONNECTION_STRING, AZURE_STORAGE_CONTAINER_NAME)
searcher = AzureAISearcher(AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX_NAME)

# ---------------------------------------------------------------
# 5. 機種型式一覧（machinelist.csv） 検索Function
# ---------------------------------------------------------------
def get_machine_list(vehicle_type, manufacturer=None, model_keyword=None):
    """
    Blobから機種リスト(model_list.json)を取得し、指定された条件でフィルタリングする。
    vehicle_typeの指定は曖昧でも、GPTがJSON内のキーに変換を試みる。
    """
    dbg(f"get_machine_list called with: vehicle_type='{vehicle_type}', manufacturer='{manufacturer}', model_keyword='{model_keyword}'")
    try:
        # 1. Blobからmodel_list.jsonを読み込む
        json_content = blob_manager.read_file("output_json/model_list.json")
        machine_json = json.loads(json_content)
        
        # 2. GPTを使用して、ユーザー入力をJSONのキーにマッピングする
        available_vehicle_types = list(machine_json.keys())
        
        prompt = f"""ユーザーが指定した建設機械の種類に最も一致するカテゴリを、以下のリストから1つだけ選んでください。
リスト: {', '.join(available_vehicle_types)}
ユーザー入力: "{vehicle_type}"
カテゴリ名のみを返してください。"""
        dbg("Prompt for vehicle type mapping:", prompt)
        
        response = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "あなたは与えられたリストの中から最も適切なカテゴリ名を返すアシスタントです。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=50
        )
        
        gpt_vehicle_type = response.choices[0].message.content.strip()
        dbg(f"GPT mapped vehicle type: '{vehicle_type}' -> '{gpt_vehicle_type}'")

        # 3. データの抽出とフィルタリング
        if gpt_vehicle_type in machine_json:
            machine_data = machine_json[gpt_vehicle_type]
            
            # フィルタリング条件がなければ、その車種の全データを返す
            if not manufacturer and not model_keyword:
                return machine_data

            results = []
            for entry in machine_data:
                # メーカーでフィルタ
                manufacturer_match = not manufacturer or entry.get("manufacturer") == manufacturer
                if not manufacturer_match:
                    continue

                # モデルキーワードでフィルタ
                if model_keyword:
                    matching_models = [m for m in entry.get("models", []) if model_keyword.lower() in m.lower()]
                    if not matching_models:
                        continue
                    # マッチしたモデルのみを含む新しいdictを作成
                    results.append({
                        "manufacturer": entry.get("manufacturer"),
                        "models": matching_models
                    })
                elif manufacturer_match: # model_keywordがなく、メーカーが一致した場合
                    results.append(entry)
            
            return results
        else:
            dbg(f"GPTが返した車種がJSON内に見つかりませんでした: {gpt_vehicle_type}")
            return []

    except Exception as e:
        dbg(f"Blobからの model_list.json の読み込みまたは処理に失敗しました: {e}")
        return []

# PDFツリー構造サイドバー等
def get_all_pdf_paths(prefix="input/") -> list[str]:
    """BlobからPDFとCSVのパス一覧を取得"""
    all_blobs = blob_manager.list_files(prefix=prefix)
    return [b for b in all_blobs if b.lower().endswith(('.pdf', '.csv'))]

def get_document_summary(json_path='Document_explanation.json'):
    """Blobからdocument_explanation.jsonを読み込む"""
    try:
        content = blob_manager.read_file(json_path)
        return json.loads(content)
    except Exception as e:
        dbg(f"Failed to read {json_path} from blob: {e}")
        return {}

hits = None
pdf_root = Path("PDF")
pdf_paths = get_all_pdf_paths()
split_pdf_paths = get_all_pdf_paths(prefix='split_pdfs/')
pdf_paths_str = "\n".join(pdf_paths)
split_pdf_paths_str = "\n".join(split_pdf_paths)
summary_data = get_document_summary()

if 'thoughts' not in st.session_state:
    st.session_state.thoughts = []

SYSTEM_INDEX_GUIDE = (
    "You are a search assistant for PDF documents related to the construction machinery industry.\n"
    "Please answer in the user's question language.\n"
    "Your primary goal is to answer the user's question by following a strict plan-verify loop.\n\n"
    "**Workflow:**\n"
    "1. **PLAN**: You must first call the `plan` function with the user's query. This function will perform a comprehensive search and return aggregated results.\n"
    "2. **VERIFY**: Next, you must call the `verify` function with the results from `plan`. This function will judge if the results are sufficient to answer the question.\n"
    "3. **LOOP or ANSWER**:\n"
    "   - **If `verify` judges the results as INSUFFICIENT**: You MUST try again. Refine your search strategy based on the initial results and the user's original request. Then, call the `plan` function again with a refined query. You must continue this PLAN -> VERIFY loop until you gather sufficient information.\n"
    "   - **If `verify` judges the results as SUFFICIENT**: Construct the final, detailed response for the user. In your answer, you must state which PDF documents were used as references.\n\n"
    "**Important Rules:**\n"
    "- Do not give up easily. The loop is designed to help you refine your search and find the correct information.\n"
    "- If, after several attempts, you still cannot find the information, you may then inform the user that the information could not be found.\n"
    "- Never fabricate an answer. All information must come from the provided search results.\n"
    "- Make your final answers detailed and easy to understand, using tables where appropriate."
)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if len(st.session_state.messages) == 0:   
    st.session_state.messages = [{"role": "system", "content": SYSTEM_INDEX_GUIDE}]

# ---------------------------------------------------------------
# 6. Function calling登録
# ---------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "plan",
            "description": (
                "Analyzes the user's query, identifies relevant documents, runs a specific search for each, and returns the aggregated results. "
                "This must be the first step in the process."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {"type": "string", "description": "The user's query to analyze"}
                },
                "required": ["messages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify",
            "description": (
                "This function is called after the 'plan' function has executed. "
                "It checks whether the aggregated search results are sufficient to meet the user's intent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {"type": "string", "description": "The message to pass in, including the aggregated search results from the plan"}
                },
                "required": ["messages"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_machine_list",
            "description": (
                "Used when the vehicle model type is not specified or when confirming the existence of a vehicle. "
                "It extracts information such as machine_type, size, and model-type."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "messages": {"type": "string", "description": "The message to pass in"}
                },
                "required": ["messages"]
            }
        }
    }
]

# ---------------------------------------------------------------
# 7. Function Callの実体ディスパッチ
# ---------------------------------------------------------------
def function_call_dispatch(call, searcher, user_input):
    if call.function.name == "plan":
        args = json.loads(call.function.arguments)
        plan_input = args.get("messages", "")
        st.session_state.thoughts.append(f"Planning based on: {plan_input}")
        st.write(f"--- Step 1: Extracting models from query ---")

        # 1. Extract machine model from user query
        model_extraction_prompt = (
            "From the user's query, extract any construction machinery model names. "
            "Return the response as a JSON object with a single key 'models' which is a list of strings. "
            "For example: {\\\"models\\\": [\\\"PC200-10\\\", \\\"PC350LC-8\\\"]}. "
            "If no models are found, return an empty list: {\\\"models\\\": []}."
        )
        model_extraction_resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": model_extraction_prompt},
                {"role": "user", "content": plan_input}
            ],
            response_format={"type": "json_object"}
        )
        extracted_models_str = model_extraction_resp.choices[0].message.content
        extracted_models_json = json.loads(extracted_models_str)
        extracted_models = extracted_models_json.get("models", [])
        st.write(f"Extracted models: {extracted_models}")

        # 2. Filter document summaries based on extracted models
        st.write(f"--- Step 2: Filtering summaries based on models ---")
        filtered_summary_data = {}
        if not extracted_models:
            st.write("No specific model extracted, using all document summaries for planning.")
            filtered_summary_data = summary_data
        else:
            st.write("Filtering documents using fuzzy matching...")
            for doc_title, data in summary_data.items():
                model_list = data.get("model list", [])
                if not model_list:
                    continue
                found_match_for_doc = False
                model_list.append("all_models")
                for extracted_model in extracted_models:
                    for model_in_list in model_list:
                        similarity_score = fuzz.partial_ratio(extracted_model.lower(), model_in_list.lower())
                        if similarity_score > 80:
                            filtered_summary_data[doc_title] = data
                            st.write(f"Found match for '{extracted_model}' in '{doc_title}' (Score: {similarity_score}). Adding to plan context.")
                            found_match_for_doc = True
                            break
                    if found_match_for_doc:
                        break
        
        st.write(f"Filtered down to {len(filtered_summary_data)} relevant document(s).")

        # 3. Create a search plan using the filtered summaries
        st.write(f"--- Step 3: Generating search plan ---")
        all_search_results = []
        if not filtered_summary_data:
            st.write("No relevant document summaries found after filtering. No search will be performed.")
        else:
            summaries_text = json.dumps(filtered_summary_data, indent=2, ensure_ascii=False)
            planning_system_prompt = (
                "You are an expert search planner. Your task is to analyze a user's query and a list of available documents with their summaries to create a search plan.\\n"
                "First, provide your reasoning in a 'thought' field, explaining which documents you selected and why, and how you constructed the queries.\\n"
                "Then, provide the 'search_plan' itself. The plan should consist of a list of search actions. Each action must specify the document to search and the optimized simple search query for that specific document.\\n"
                "Your output MUST be a JSON object with two keys: `thought` (a string with your reasoning) and `search_plan` (a list of search action objects).\\n"
                "Example output: {\\\\\"thought\\\\\": \\\\\"The user is asking about X. Document A seems relevant because its summary mentions X. I will craft a query for Document A to find details about X...\\\\\", \\\\\"search_plan\\\\\": [{\\\\\"document_to_search\\\\\": \\\\\"doc1.pdf\\\\\", \\\\\"search_query\\\\\": \\\\\"specific question about doc1\\\\\"}]}"
            )

            st.write("Generating search plan from filtered summaries...")
            resp_plan = client.chat.completions.create(
                model=CHAT_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": planning_system_prompt},
                    {"role": "user", "content": f"User Query: \\\"{plan_input}\\\"\\\\n\\\\nAvailable Documents and Summaries:\\\\n{summaries_text}"}
                ],
                response_format={"type": "json_object"},
            )
            
            plan_content_str = resp_plan.choices[0].message.content
            plan_content_json = json.loads(plan_content_str)
            thought = plan_content_json.get("thought", "No thought process provided.")
            search_plan = plan_content_json.get("search_plan", [])

            st.write("Planner's Thought Process:")
            st.info(thought)
            st.session_state.thoughts.append(f"Planner's Thought: {thought}")

            st.write("Search Plan Created:")
            st.json(search_plan)

            # 4. Execute the search plan
            if not search_plan:
                st.write("The generated plan is empty. No documents will be searched.")
            else:
                st.write("--- Step 4: Executing search plan ---")
                for search_action in search_plan:
                    doc_title = search_action.get("document_to_search")
                    search_query = search_action.get("search_query")

                    if not doc_title or not search_query:
                        st.write(f"Skipping invalid search action: {search_action}")
                        continue

                    st.write(f"--- Searching in: {doc_title} ---")
                    st.write(f"Query: {search_query}")
                    
                    search_results = searcher.search(
                        query=search_query, 
                        filter_expression=f"original_path eq '{doc_title}'", 
                        top_k=3
                    )
                    st.write(f"Found {len(search_results)} results.")
                    all_search_results.extend(search_results)

        st.write("--- Plan execution complete. Returning all results. ---")
        return {
            "role": "tool",
            "name": "plan",
            "tool_call_id": call.id,
            "content": json.dumps(all_search_results, ensure_ascii=False),
        }
    elif call.function.name == "search_documents":
        args = json.loads(call.function.arguments)
        result = searcher.search(**args)
        return {
            "role": "tool",
            "name": "search_documents",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        }
    elif call.function.name == "get_machine_list":
        args = json.loads(call.function.arguments)
        
        # For debugging, show what arguments the agent provided
        thought = f"Tool call: get_machine_list with args: {args}"
        st.session_state.thoughts.append(thought)
        st.write(thought)

        # Execute the function with the arguments provided by the agent
        # The agent might pass the natural language query in different ways.
        # We try to find it from common keys like 'vehicle_type', 'messages', or 'query'.
        # The get_machine_list function is designed to handle natural language input for vehicle_type.
        natural_language_query = args.get("vehicle_type") or args.get("messages") or args.get("query")
        if not natural_language_query and isinstance(args, dict) and args:
            # As a fallback, use the first value in the args dictionary if it's a string.
            first_value = next(iter(args.values()))
            if isinstance(first_value, str):
                natural_language_query = first_value

        machine_list_result = get_machine_list(
            vehicle_type=natural_language_query,
            manufacturer=args.get("manufacturer"),
            model_keyword=args.get("model_keyword")
        )

        # For debugging, show the result
        st.session_state.thoughts.append(str(machine_list_result))
        st.write('machine list result:')
        st.write(machine_list_result)

        return {
            "role": "tool",
            "name": "get_machine_list",
            "tool_call_id": call.id,
            "content": json.dumps(machine_list_result, ensure_ascii=False),
        }
    elif call.function.name == "verify":
        args = json.loads(call.function.arguments)
        verify_input = args.get("messages", "")
        st.write('message from top agent')
        st.write(verify_input)

        resp_verify = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a verification assistant. Your task is to check if the search results are sufficient to answer the user's question.\n"
                        "Based on your analysis, you must construct a JSON object with three fields: 'thought' (your reasoning), 'answer_language' (the language of the answer), and 'judge' (a boolean indicating whether the results are sufficient).\n"
                        "If the results are insufficient, you may need to call 'plan' again with a refined query.\n"
                        "If the results are sufficient, you can construct the final response. In the final answer, indicate which PDF(s) were referenced to generate the response.\n"
                        "The output must be a JSON object with the three fields mentioned above."
                    )
                },
                {
                    "role": "user",
                    "content": f"Here are the search results, please verify them:\n\n{verify_input}"
                },
            ],
            response_format={"type": "json_object"},
        )
        verify_content = resp_verify.choices[0].message.content
        st.session_state.thoughts.append(json.loads(verify_content)['thought'])
        st.write('verify:')
        st.write(json.loads(verify_content)['thought'])

        return {
            "role": "tool",
            "name": "verify",
            "tool_call_id": call.id,
            "content": verify_content,
        }

    return None

# ---------------------------------------------------------------
# 8. Multi-step function-callingの主処理
# ---------------------------------------------------------------
def multi_step_chat(user_input):
    st.session_state.pdf_paths = []  # ヒットしたPDFパスのクリア
    st.session_state.messages.append({"role": "user", "content": user_input})
    while True:
        resp = client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=st.session_state.messages,
            tools=tools,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        st.session_state.messages.append(msg.model_dump(exclude_none=True))

        if msg.tool_calls:
            for call in msg.tool_calls:
                tool_msg = function_call_dispatch(call, searcher, user_input)
                if tool_msg is not None:
                    if call.function.name == "search_documents":
                        hits = json.loads(tool_msg["content"])
                        if hits:
                            for hit in hits:
                                if pdf_path := hit.get("pdf_path"):
                                    st.session_state.pdf_paths.append([pdf_path, round(hit.get("score", 0), 3)])
                    st.session_state.messages.append(tool_msg)
            
            # Continue the loop to let the agent process the tool result
            continue
        
        # If no tool call, it's the final answer.
        break

# ---------------------------------------------------------------
# 9. Streamlit UI部（PDFリストUIは既存のまま流用）
# ---------------------------------------------------------------
st.set_page_config(page_title="Sales genius")
st.title("Sales genius")

def build_tree(paths: list[str], root: Path) -> dict:
    tree: dict[str, dict] = {}
    for p in paths:
        rel = Path(p).relative_to(root)
        node = tree
        for part in rel.parts:
            node = node.setdefault(part, {})
    return tree

def render_tree(node: dict, parent_expander=None):
    for name, child in node.items():
        if child:
            exp = (parent_expander or st.sidebar).expander(name, expanded=False)
            render_tree(child, exp)
        else:
            (parent_expander or st.sidebar).write(name)

with st.sidebar:
    if st.button("New chat"):
        st.session_state.messages = []
        st.session_state.pdf_paths = []
        st.session_state.thoughts = []
    st.title("Database list")
    tree = build_tree(get_all_pdf_paths(), Path("input"))
    render_tree(tree)



if 'pdf_paths' not in st.session_state:
    st.session_state.pdf_paths = []

def get_binary_file_downloader_html(bin_file: str, label: str) -> str:
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{label}</a>'

def display_download_link(download_file: str, title: str, percentage: float) -> None:
    st.markdown(get_binary_file_downloader_html(download_file, f"{title} : {percentage}"), unsafe_allow_html=True)

def display_new_messages():
    for msg in st.session_state.messages:
        if msg["role"] == "system" or msg['role']=='tool' or msg.get('content')==None:
            continue
        else:
            st.chat_message(msg["role"]).write(msg["content"])

def voice_save(audio_data):
    if audio_data:
        # 録音された音声データをファイルとして保存
        os.makedirs('voice',exist_ok=True)
        audio_file_path = os.path.join('voice', 'recorded_audio.wav')  # 保存するファイル名
        with open(audio_file_path, 'wb') as f:
            f.write(audio_data)


def GPT_transcript():
    try:
        # 音声データをOpenAI APIに送信
        audio_file= open(os.path.join('voice','recorded_audio.wav') , "rb")
        print('[DEBUG] audio loaded')
        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe", 
            file=audio_file,
            )
        print(transcription)
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
    return transcription.text

from audio_recorder_streamlit import audio_recorder
from thefuzz import fuzz
# ---------------------------------------------------------------
# 10. ユーザー入力・実行
# ---------------------------------------------------------------
# Record audio and get transcription
voice_data = audio_recorder(
    text='input question by voice',
    icon_size="2x",
    key="voice",
    pause_threshold=120
)
transcription = None
if voice_data:
    voice_save(voice_data)
    transcription = GPT_transcript()

if transcription:
    # Show the transcribed text in chat for confirmation
    st.write(transcription)
    # Offer to send as-is or edit
    if st.button("Send as is"):
        multi_step_chat(transcription)
else:
    # Fallback to manual input
    user_input = st.chat_input("Enter the question...")
    if user_input:
        multi_step_chat(user_input)

display_new_messages()
disp_list = []
for p, score in st.session_state.pdf_paths:
    if p not in disp_list:
        disp_list.append(p)
        rel_path = os.path.relpath(p, start="split_pdfs")
        display_download_link(p, rel_path, score)
