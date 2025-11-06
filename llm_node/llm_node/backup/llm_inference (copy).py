# -*- coding: utf-8 -*-
"""
tool_chat_engine.py (RAG-Mode + LLM-Tool, JSONL-backed v5.6)
- 체인 없이 RAG: Retriever → Llama 직접 프롬프트
- 모드: START | UP | DOWN | DELIVER | RETURN | DISASSEMBLE | MEASURE | FINISH | INFO
- DELIVER: tool + target 동시 추출, 누락 시 공간표현 보정
- RETURN: last_tool 폴백
- explicit_tool 있어도 DELIVER 타깃 보강 수행
- INFO: 동작 호출 없이 '필요 도구'만 식별하고 control="INFO"로 전달
- plan_action 포함, TTS 특수토큰 제거
"""

import os, re, json, unicodedata, warnings
from textwrap import dedent
from typing import Optional, Dict, Any, List, Tuple

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from pydantic import Field, model_validator
from llama_cpp import Llama
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.embeddings import Embeddings
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ToolChatEngine:
    CANDIDATES = {"START", "UP", "DOWN", "DELIVER", "RETURN", "DISASSEMBLE", "MEASURE", "FINISH", "INFO"}
    _MODE_RE = re.compile(r"\b(START|UP|DOWN|DELIVER|RETURN|DISASSEMBLE|MEASURE|FINISH|INFO)\b", re.IGNORECASE)

    ALLOWED_TOOLS = {
        "스탠드": ["스탠드","조명","작업등","스탠드 조명","스탠드불","스탠드 불","lamp"],
        "M3 나사": ["M3 나사","M3 볼트","작은 나사","볼트","나사","m3", "엠쓰리", "엠쓰리 나사"],
        "니퍼": [
            "니퍼","니뻐","니빠","nipper",
            "롱노즈","롱노즈 플라이어","롱노우즈 플라이어","롱 플라이어","롱플라이어",
            "needle nose","needle-nose","long nose","long-nose"
        ],
        "커터": ["커터","wire cutter","와이어 커터","와이어커터"],
        "와이어 스트리퍼": ["와이어 스트리퍼","스트리퍼","와이어스트리퍼","피복 스트리퍼","피복제거기","wire stripper"],
    }

    DIRECTION_KEYWORDS = {
        "front": ["앞", "앞쪽", "전방", "앞에", "front", "forward"],
        "back":  ["뒤", "뒤쪽", "후방", "뒤에", "rear", "back"],
        "left":  ["왼쪽", "좌", "좌측", "왼편", "left"],
        "right": ["오른쪽", "우", "우측", "오른편", "right"],
    }

    DEFAULT_SCENARIO = dedent("""
    [시나리오 지침: 모드 정의와 예시]
    - START: 작업 시작. "시작하자", "준비해", "작업등 켜"
    - UP: 스탠드 상승/밝게. "스탠드 올려", "조명 높여"
    - DOWN: 스탠드 하강/어둡게. "스탠드 내려", "조명 낮춰"
    - DELIVER: 공구/부품 전달. "니퍼 가져와", "M3 나사 좀"
    - RETURN: 공구 반납. "니퍼 원위치", "도구 정리해"
    - MEASURE: 배선/전류/측정. "배선 점검해줘", "전류 흐르는지 검사해줘", "검류기로 측정"
    - DISASSEMBLE: 분해/볼트 풀기. "나사 풀어", "분해해"
    - FINISH: 종료/소등. "끝내자", "작업 종료", "불 꺼"
    - INFO: 도구에 대한 설명/선택을 묻는 질의. 실제 동작은 하지 않고 '필요 도구'만 식별한다.
    - 규칙: 여러 지시가 있으면 '마지막 명령'을 우선.
    """)

    _KO_BOUND    = r'(?<![가-힣A-Za-z0-9])'
    _KO_BOUND_T  = r'(?![가-힣A-Za-z0-9])'
    _POST_SIMPLE = r'(?:를|을|은|는|이|가|과|와)?'  # 간단 조사 허용

    _RIPPER_ONLY = re.compile(
        rf'(?<!스트){_KO_BOUND}리퍼{_POST_SIMPLE}{_KO_BOUND_T}'
    )
    class C:
        GREEN, YELLOW, RED, CYAN, END = '\033[92m', '\033[93m', '\033[91m', '\033[96m', '\033[0m'

    class SimpleTfidfRetriever(BaseRetriever):
        docs: List[Document]
        k: int = 3
        tfidf_mat: np.ndarray = Field(default=None, repr=False)
        idf: np.ndarray = Field(default=None, repr=False)
        vocab: dict = Field(default=None, repr=False)

        class Config:
            arbitrary_types_allowed = True

        @model_validator(mode='after')
        def build_tfidf(self) -> 'ToolChatEngine.SimpleTfidfRetriever':
            if self.tfidf_mat is not None:
                return self
            tokens_per_doc = [ToolChatEngine._tokenize(d.page_content) for d in self.docs]
            vocab = {}
            for toks in tokens_per_doc:
                for t in toks:
                    if t not in vocab:
                        vocab[t] = len(vocab)
            self.vocab = vocab
            V, N = len(vocab), len(self.docs)
            df = np.zeros(V, dtype=np.float32)
            for toks in tokens_per_doc:
                for t in set(toks):
                    if t in vocab:
                        df[vocab[t]] += 1.0
            self.idf = np.log((N + 1) / (df + 1)) + 1.0
            self.tfidf_mat = np.zeros((N, V), dtype=np.float32)
            for i, toks in enumerate(tokens_per_doc):
                if not toks:
                    continue
                tf = {t: toks.count(t) for t in set(toks)}
                vec = np.zeros(V, dtype=np.float32)
                for t, c in tf.items():
                    j = self.vocab[t]
                    vec[j] = (c / len(toks)) * self.idf[j]
                norm = np.linalg.norm(vec)
                self.tfidf_mat[i, :] = vec / (norm + 1e-9)
            return self

        def _get_relevant_documents(self, query: str) -> List[Document]:
            q_toks = ToolChatEngine._tokenize(query)
            if not q_toks or not self.vocab:
                return self.docs[:self.k]
            V = len(self.vocab)
            q_vec = np.zeros(V, dtype=np.float32)
            tf = {t: q_toks.count(t) for t in set(q_toks) if t in self.vocab}
            if not tf:
                return self.docs[:self.k]
            for t, c in tf.items():
                j = self.vocab[t]
                q_vec[j] = (c / len(q_toks)) * self.idf[j]
            q_norm = np.linalg.norm(q_vec)
            if q_norm < 1e-9:
                return self.docs[:self.k]
            q_vec = q_vec / q_norm
            sims = self.tfidf_mat @ q_vec
            top_idx = np.argsort(-sims)[:self.k]
            return [self.docs[i] for i in top_idx]

    @staticmethod
    def _normalize_kr(s: str) -> str:
        s = unicodedata.normalize('NFKC', s or "")
        s = s.lower()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    _WORD_RE = re.compile(r"[A-Za-z0-9]+|[가-힣]+")
    @staticmethod
    def _tokenize(text: str):
        return ToolChatEngine._WORD_RE.findall((text or "").lower())

    # ───────── init ─────────
    def __init__(self,
                 model_path: str = "/home/liw/sllm/models/gemma3-q4_k_m_budda.gguf",
                 mode_rag_jsonl_path: Optional[str] = None,
                 rag_mode_text: Optional[str] = None,
                 faiss_index_path: str = "./faiss_index_mode",
                 use_gpu_llama: bool = True,
                 verbose: bool = True):
        self.verbose = verbose
        self.model_path = model_path
        self.use_gpu_llama = use_gpu_llama

        if torch.cuda.is_available() and self.use_gpu_llama:
            if self.verbose:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"{self.C.CYAN}[*] GPU 감지({gpu_count}), 주 GPU: {gpu_name}{self.C.END}")
                print(f"{self.C.GREEN}[*] GPU 오프로드 활성화{self.C.END}")
            self.n_gpu_layers = -1
        else:
            if self.verbose:
                print(f"{self.C.YELLOW}[*] GPU 미사용 → CPU 모드{self.C.END}")
            self.n_gpu_layers = 0

        self.mode_rag_jsonl_path = mode_rag_jsonl_path
        self.faiss_index_path = faiss_index_path
        self.session_state: Dict[str, Any] = {"last_tool": None, "last_mode": None, "last_object": None}

        if self.verbose:
            print(f"[*] GGUF 모델 로드: {self.model_path}", flush=True)

        if self.n_gpu_layers > 0:
            n_ctx, n_batch, n_threads = 4096, 2048, 4
        else:
            n_ctx, n_batch, n_threads = 2048, 1024, min(os.cpu_count() or 4, 8)

        self.llm_model = Llama(
            model_path=self.model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=self.n_gpu_layers,
            seed=0,
            verbose=False,
            use_mlock=True,
            use_mmap=True,
        )

        self.embeddings = self._load_embedding_model()
        self.mode_retriever, self.mode_template = self._setup_mode_rag(
            scenario_text=(rag_mode_text or self.DEFAULT_SCENARIO),
            jsonl_path=self.mode_rag_jsonl_path
        )

    def __del__(self):
        try:
            if hasattr(self, 'llm_model') and self.llm_model is not None:
                try:
                    if hasattr(self.llm_model, 'close'):
                        self.llm_model.close()
                finally:
                    self.llm_model = None
        except Exception:
            pass

    # ───── embedding / RAG ─────
    def _load_embedding_model(self) -> Optional[Embeddings]:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            if self.verbose:
                print(f"{self.C.CYAN}[*] 임베딩 로드(jhgan/ko-sbert-nli){self.C.END}", flush=True)
            return HuggingFaceEmbeddings(
                model_name='jhgan/ko-sbert-nli',
                model_kwargs={'device': device},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            if self.verbose:
                print(f"{self.C.YELLOW}[!] 임베딩 로드 실패: {e}{self.C.END}")
            return None

    def _load_mode_docs_from_jsonl(self, jsonl_path: str) -> List[Document]:
        docs: List[Document] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                docs.append(Document(page_content=obj["page_content"],
                                     metadata={"mode": obj.get("mode", "UNKNOWN"),
                                               "type": obj.get("type", "unknown")}))
        return docs

    def _setup_mode_rag(self, scenario_text: str, jsonl_path: Optional[str]):
        if self.verbose:
            print("[*] Mode RAG 준비...", flush=True)

        retriever = None
        if self.embeddings:
            if os.path.exists(self.faiss_index_path):
                try:
                    if self.verbose:
                        print(f"{self.C.GREEN}[*] FAISS 인덱스 로드: {self.faiss_index_path}{self.C.END}")
                    vectorstore = FAISS.load_local(self.faiss_index_path, self.embeddings, allow_dangerous_deserialization=True)
                    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
                except Exception as e:
                    if self.verbose:
                        print(f"{self.C.YELLOW}[!] FAISS 로드 실패: {e}{self.C.END}")

            if retriever is None:
                if self.verbose:
                    print(f"{self.C.CYAN}[*] 새 FAISS 인덱스 생성{self.C.END}")
                if jsonl_path and os.path.isfile(jsonl_path):
                    docs = self._load_mode_docs_from_jsonl(jsonl_path)
                else:
                    docs = [Document(page_content=scenario_text, metadata={"section": "scenario"})]
                vectorstore = FAISS.from_documents(docs, self.embeddings)
                vectorstore.save_local(self.faiss_index_path)
                retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

        if retriever is None:
            if self.verbose:
                print(f"{self.C.YELLOW}[!] 임베딩 실패 → TF-IDF 폴백{self.C.END}")
            docs = self._load_mode_docs_from_jsonl(jsonl_path) if jsonl_path and os.path.isfile(jsonl_path) else [Document(page_content=scenario_text)]
            retriever = self.SimpleTfidfRetriever(docs=docs, k=5)

        mode_template = (
            "당신은 사용자 발화로부터 '모드'만 결정하는 시스템입니다.\n"
            "가능한 모드: START | UP | DOWN | DELIVER | RETURN | DISASSEMBLE | MEASURE | FINISH | INFO\n"
            "규칙: 문장에 여러 동작이 있으면 마지막 명령을 우선합니다.\n"
            "아래 Context와 질문을 보고 가장 적합한 모드 하나만 골라 출력하세요.\n"
            "출력은 정확히 'Mode: <모드명>' 한 줄.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        return retriever, mode_template

    def rag_infer_mode(self, user_text: str) -> Optional[str]:
        if not self.mode_retriever:
            return None
        try:
            try:
                docs = self.mode_retriever.invoke(user_text)
            except Exception:
                docs = self.mode_retriever.invoke(user_text)
            context = "\n\n".join(d.page_content for d in docs)[:4000]
            prompt = self.mode_template.format(context=context, question=user_text)
            raw = self.llm_model.create_completion(
                prompt=prompt, max_tokens=8, temperature=0.0, stop=["\n"]
            )['choices'][0]['text']
            m = self._MODE_RE.search(raw or "")
            return m.group(1).upper() if m else None
        except Exception:
            return None

    # ───── canon/추출 유틸 ─────
    def _canon_from_value(self, value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        txt = self._normalize_kr(value)
        for canon, aliases in self.ALLOWED_TOOLS.items():
            if self._normalize_kr(canon) == txt:
                return canon
            for a in aliases:
                if self._normalize_kr(a) == txt:
                    return canon
        return None

    _SPATIAL_TOKENS = r"(옆|옆에|옆에다|근처|앞|뒤|위|아래|우측|좌측|오른쪽|왼쪽)"
    _POSTPOSITION = r"(에다|에|으로|로|에다가)?"
    def _clean_target_phrase(self, s: str) -> str:
        s = (s or "").strip()
        s = re.sub(rf"\s*{self._SPATIAL_TOKENS}{self._POSTPOSITION}$", "", s)
        s = re.sub(r"(을|를|은|는|이|가|와|과)$", "", s)
        return s.strip()

    _SPATIAL = r"(옆|옆에|옆에다|근처|앞|뒤|위|아래|우측|좌측|오른쪽|왼쪽)"
    def _extract_target_from_spatial(self, text: str, exclude: Optional[str] = None) -> Optional[str]:
        t = self._normalize_kr(text)
        if re.search(rf'{self._RIPPER_ONLY.pattern}\s*{self._SPATIAL}', t) or \
           re.search(rf'{self._SPATIAL}\s*{self._RIPPER_ONLY.pattern}', t):
            return "니퍼"

        alias_pairs = []
        for canon, aliases in self.ALLOWED_TOOLS.items():
            if exclude and canon == exclude:
                continue
            for name in [canon] + aliases:
                n = self._normalize_kr(name)
                alias_pairs.append((canon, n))
        alias_pairs.sort(key=lambda x: len(x[1]), reverse=True)

        spatial = r"(옆|옆에|옆에다|근처|앞|뒤|위|아래|우측|좌측|오른쪽|왼쪽)"
        for canon, n in alias_pairs:
            if not n: continue
            if re.search(rf"{re.escape(n)}\s*{spatial}", t):
                return canon
            if re.search(rf"{spatial}\s*{re.escape(n)}", t):
                return canon
        return None

    def canonicalize_tool(self, user_text: str) -> Optional[str]:
        txt = self._normalize_kr(user_text)

        hits = []
        if self._RIPPER_ONLY.search(txt):
            hits.append("니퍼")

        alias_pairs = []
        for canon, alias_list in self.ALLOWED_TOOLS.items():
            for a in [canon] + alias_list:
                n = self._normalize_kr(a)
                if n:
                    alias_pairs.append((canon, n))
        alias_pairs.sort(key=lambda x: len(x[1]), reverse=True)

        for canon, n in alias_pairs:
            if n in txt and canon not in hits:
                hits.append(canon)

        if len(hits) >= 2 and "M3 나사" in hits:
            return "M3 나사"
        return hits[0] if hits else None

    # ───── LLM 도구/타깃 추출 ─────
    def llm_pick_tool(self, user_text: str) -> Optional[str]:
        try:
            choices = " | ".join(sorted(self.ALLOWED_TOOLS.keys()))
            prompt = (
                "역할: 아래 발화에 가장 적합한 공구를 선택지에서 하나만 골라 정식명으로 답하라.\n"
                f"선택지: {choices}\n"
                "출력 형식: {tool: <정식명 또는 NONE>}\n"
                f"문장: {user_text}\n응답: "
            )
            raw = self.llm_model.create_completion(prompt=prompt, max_tokens=16, temperature=0.0, stop=["}"])['choices'][0]['text']
            for canon in self.ALLOWED_TOOLS.keys():
                if canon in raw:
                    return canon
            if "NONE" in raw.upper():
                return None
        except Exception:
            pass
        return None

    def llm_extract_tool_and_target(self, user_text: str) -> Tuple[Optional[str], Optional[str]]:
        def _extract_json_obj(raw: str) -> dict:
            _JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
            m = _JSON_OBJ_RE.search(raw or "")
            if not m:
                raise ValueError("no json object")
            frag = m.group(0)
            frag = frag.replace("'", '"')
            frag = re.sub(r",\s*([}\]])", r"\1", frag)
            return json.loads(frag)

        try:
            tool_list = " | ".join(sorted(self.ALLOWED_TOOLS.keys()))
            prompt = (
                "역할: 문장에서 '전달할 도구'와 '목표 대상'을 추출한다.\n"
                "규칙: <도구>는 반드시 [도구 목록]의 정식 명칭만 사용. 없으면 NONE.\n"
                "규칙: <대상>은 반드시 [대상 목록]의 정식 명칭만 사용. 없으면 NONE.\n"
                "규칙: '<X>를 <Y> <방향>에' 패턴이면 X=<도구>, Y=<대상>.\n"
                "규칙: '<X>를 …' 패턴이면 X=도구, target=NONE.\n"
                '출력: {"tool":"<도구|NONE>","target":"<대상|NONE>"}\n'
                f"[도구 목록]: {tool_list}\n"
                f"[대상 목록]: {tool_list}\n"
                f"문장: {user_text}\n"
                "응답: "
            )
            raw_json = self.llm_model.create_completion(
                prompt=prompt, max_tokens=64, temperature=0.0, stop=["}"]
            )['choices'][0]['text'] + "}"
            result = _extract_json_obj(raw_json)
            tool_raw   = (result.get("tool")   or "").strip()
            target_raw = (result.get("target") or "").strip()
            target_raw = self._clean_target_phrase(target_raw)

            final_tool   = self._canon_from_value(tool_raw)
            final_target = self._canon_from_value(target_raw)
            if final_target is None:
                final_target = self._extract_target_from_spatial(user_text, exclude=final_tool)

            return final_tool, final_target
        except Exception:
            return self.canonicalize_tool(user_text), self._extract_target_from_spatial(user_text)

    # ───── 방향 추출 ─────
    def extract_direction(self, user_text: str) -> Optional[str]:
        txt = self._normalize_kr(user_text)
        for direction, keywords in self.DIRECTION_KEYWORDS.items():
            for kw in keywords:
                if kw in txt:
                    return direction
        return None

    # ───── 모드/명령 파이프라인 ─────
    def parse_and_infer(self, user_text: str) -> Dict[str, Any]:
        mode: Optional[str] = None
        tool: Optional[str] = None
        target_object: Optional[str] = None
        direction: Optional[str] = None
        control_string: Optional[str] = None

        mode = self.rag_infer_mode(user_text)
        if not mode:
            system_instructions = (
                "사용자 한국어 명령을 9모드(START, UP, DOWN, DELIVER, RETURN, DISASSEMBLE, MEASURE, FINISH, INFO) 중 하나로 분류하라.\n"
                "여러 지시가 섞이면 마지막 명령을 우선한다.\n"
                "출력은 영어 대문자 모드명 한 단어."
            )
            prompt_to_send = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            try:
                raw = self.llm_model.create_completion(
                    prompt=prompt_to_send, max_tokens=8, temperature=0.0, stop=["\n"]
                )['choices'][0]['text']
                m = self._MODE_RE.search(raw or "")
                mode = m.group(1).upper() if m else None
            except Exception:
                mode = None

        # ===== INFO 전용 처리: 동작 호출 금지, 도구만 식별 =====
        if mode == "INFO":
            tool_guess = self.canonicalize_tool(user_text) or self.llm_pick_tool(user_text)
            return {
                "intent": "질의",
                "mode": "INFO",
                "tool_final": (tool_guess or "NONE"),
                "target_object": "NONE",
                "direction": "NONE",
                "control_string": "INFO",
                "ask_back": bool(tool_guess is None),
                "question_only": False,
                "should_call_function": False,   # ★ 로봇 동작 호출 금지
                "function_name": "",
                "function_args": {},
            }

        explicit_tool = self.canonicalize_tool(user_text)
        _, lt = self.llm_extract_tool_and_target(user_text)

        if mode == "DELIVER":
            if explicit_tool:
                tool = explicit_tool
                target_object = self._extract_target_from_spatial(user_text, exclude=tool)
                if target_object is None:
                    _, lt = self.llm_extract_tool_and_target(user_text)
                    if lt:
                        target_object = lt
            else:
                tool, target_object = self.llm_extract_tool_and_target(user_text)
                if target_object is None:
                    target_object = self._extract_target_from_spatial(user_text, exclude=tool)

        elif mode == "RETURN":
            last = self.session_state.get("last_tool")
            tool = self.llm_pick_tool(user_text) or (last if last in self.ALLOWED_TOOLS else None)

        else:
            if explicit_tool:
                tool = explicit_tool

        if mode == "DELIVER":
            direction = self.extract_direction(user_text)

        if self.verbose:
            print(f"{self.C.CYAN}[결과] mode={mode}, tool={tool}, target_object={target_object}, direction={direction}{self.C.END}")

        ask_back = bool(mode in {"DELIVER", "RETURN"} and tool is None)

        should_call = False
        function_name = ""
        function_args: Dict[str, Any] = {}

        if not ask_back and mode:
            if mode == "START":
                should_call, function_name, control_string = True, "start", "START"
            elif mode == "FINISH":
                should_call, function_name, control_string = True, "finish", "FINISH"
            elif mode == "DISASSEMBLE":
                should_call, function_name, control_string = True, "disassemble", "DISASSEMBLE"
            elif mode == "UP":
                should_call, function_name, control_string = True, "stand_up", "UP"
            elif mode == "MEASURE":
                should_call, function_name, control_string = True, "measure", "MEASURE"
            elif mode == "DOWN":
                should_call, function_name, control_string = True, "stand_down", "DOWN"
            elif mode == "DELIVER" and tool:
                should_call, function_name = True, "deliver_tool"
                function_args = {"tool": tool, "target": (target_object or "user"), "note": ""}
                parts = [tool, "DELIVER"]
                if target_object:
                    parts.append(target_object)
                if direction:
                    parts.append(direction)
            elif mode == "RETURN":
                should_call, function_name = True, "return_tool"
                function_args = {"tool": tool}
                tool = "NONE"
                control_string = f"RETURN"

        if should_call and function_name == "deliver_tool" and tool:
            self.session_state["last_tool"] = tool
            self.session_state["last_mode"] = "DELIVER"

        return {
            "intent": "명령",
            "mode": mode or "NONE",
            "tool_final": tool or "NONE",
            "target_object": target_object or "NONE",
            "direction": direction or "NONE",
            "control_string": control_string or "",
            "ask_back": bool(ask_back),
            "question_only": False,
            "should_call_function": bool(should_call),
            "function_name": function_name,
            "function_args": function_args,
        }

    # ───── plan_action ─────
    def plan_action(self, parsed: Dict[str, Any], raw_text: str = "") -> Dict[str, Any]:
        mode = parsed.get("mode") or "NONE"
        tool = parsed.get("tool_final") or "NONE"
        target_object = parsed.get("target_object") or "NONE"
        direction = parsed.get("direction") or "NONE"
        ask = bool(parsed.get("ask_back"))
        should_call = bool(parsed.get("should_call_function"))
        fn = parsed.get("function_name") or ""
        args = parsed.get("function_args") or {}
        control_string = parsed.get("control_string") or ""

        if mode == "DELIVER" and direction == "NONE" and raw_text:
            direction = self.extract_direction(raw_text) or "NONE"

        confirm_text = ""
        if ask:
            if mode == "DELIVER":
                confirm_text = "어떤 공구/부품이 필요하신가요? (예: 스탠드/M3 나사/니퍼/커터/와이어 스트리퍼)"
            elif mode == "RETURN":
                confirm_text = "반납할 공구가 무엇인가요?"
            elif mode == "INFO":
                confirm_text = "어떤 공구를 말씀하시는지 알려주세요."

        if not control_string and not ask:
            if mode == "DELIVER" and tool != "NONE":
                base = f"{tool}+DELIVER"
                if target_object and target_object != "NONE":
                    control_string = f"{base}+{target_object}"
                elif direction and direction != "NONE":
                    control_string = f"{base}+{direction}"
                else:
                    control_string = base
            elif mode == "RETURN":
                control_string = f"RETURN"
            elif mode in {"START", "UP", "DOWN", "DISASSEMBLE", "MEASURE", "FINISH"}:
                control_string = mode
            elif mode == "INFO":
                control_string = "INFO"

        return {
            "mode": mode,
            "tool_final": tool,
            "target_object": target_object,
            "direction": direction,
            "control_string": control_string,
            "should_call_function": should_call,
            "function_name": fn,
            "function_args": args,
            "ask_back": ask,
            "confirm_text": confirm_text,
        }

    # ───── NLG ─────
    @staticmethod
    def _strip_special(s: str) -> str:
        if not s:
            return s
        s = re.sub(r"<\|[^>]+?\|>", "", s)
        return s.strip()

    def generate_speech(self, task_description: str, style: str = "친절하고 상냥하게") -> str:
        system_prompt = f"""
        당신은 사용자를 돕는 로봇의 친절하고 상냥한 목소리입니다. 사용자의 명령을 확인하고 로봇이 무엇을 할 것인지 자연스럽게 대답해주세요.

        [지침]
        - 문체: {style}
        - 절대로 "알겠습니다" 또는 "네" 라고만 단답형으로 대답하지 마세요.
        - 항상 완전한 문장으로 사용자의 명령을 자연스럽게 복창하며 확인해주세요.

        [좋은 답변의 예시]
        - (작업: START) -> "네, 작업을 시작합니다."
        - (작업: DELIVER, 도구: 커터) -> "네, 커터를 바로 가져다 드릴게요!"
        - (작업: DISASSEMBLE) -> "네, 분해 작업을 시작하겠습니다."
        - (작업: UP) -> "스탠드를 위로 올릴게요. 잠시만요."
        - (작업: RETURN) -> "작업대를 정리하겠습니다."
        - (작업: FINISH) -> "네, 작업을 종료합니다. 수고하셨습니다!"
        - (작업: DELIVER, 도구: M3 나사, 타깃: 니퍼, 방향: left) -> "네, 엠쓰리 나사를 니퍼 왼쪽에 가져다 드릴게요!"
        - (작업: MEASURE) -> "배선 상태를 검류기로 측정하겠습니다."
        - (작업: INFO, 도구: 니퍼) -> "필요한 공구는 니퍼입니다."
        이제 아래의 [현재 수행할 작업]에 대해 가장 적절하고 자연스러운 대답을 한 문장으로 생성해주세요.
        """
        user_prompt = f"[현재 수행할 작업]: {task_description}"
        prompt_to_send = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        try:
            response = self.llm_model.create_completion(
                prompt=prompt_to_send,
                max_tokens=50,
                temperature=0.0,
                stop=["\n", "<|eot_id|>", ".", "!", "?", "요.", "다."],
                top_p=0.9,
                repeat_penalty=1.15,
            )
            speech = response['choices'][0]['text'].strip()
            if speech.startswith(("-", "(")):
                speech = "네, 알겠습니다. 바로 시작하겠습니다."
            return speech if speech else "알겠습니다. 작업을 시작하겠습니다."
        except Exception:
            if self.verbose:
                print(f"{self.C.RED}[!] generate_speech 실패{self.C.END}")
            return "알겠습니다. 지시에 따르겠습니다."
