# -*- coding: utf-8 -*-
"""
tool_chat_engine.py (RAG-Mode + LLM-Tool, v5.11 - Super-Simplified + TTS)
- 모드: START | DELIVER | RETURN | FINISH | INFO | FOCUS (제한됨)
- INFO: 도구 질의
- FOCUS: 구역(Section) 조명
- parse_and_infer가 {"mode", "tool_final", "section"} 3개 키만 반환
- section 미언급 시 기본값 1
- generate_speech (TTS) 함수 복구
- target, direction, plan_action 등 부가 로직 제거
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
    # 6개 모드
    CANDIDATES = {"START", "DELIVER", "RETURN", "FINISH", "INFO", "FOCUS"}
    _MODE_RE = re.compile(r"\b(START|DELIVER|RETURN|FINISH|INFO|FOCUS)\b", re.IGNORECASE)

    # 한국어 canonical 이름 기준 (색 키워드는 제거, 색 처리는 LLM 프롬프트에서만)
    ALLOWED_TOOLS = {
        "스탠드": [
            "스탠드", "조명", "작업등", "스탠드 조명", "스탠드불", "스탠드 불", "lamp"
        ],
        "M3 나사": [
            "M3 나사", "M3 볼트", "작은 나사", "볼트", "나사", "m3", "엠쓰리", "엠쓰리 나사"
        ],
        "니퍼": [
            "니퍼", "니뻐", "니빠", "nipper",
            "롱노즈", "롱노즈 플라이어", "롱노우즈 플라이어", "롱 플라이어", "롱플라이어",
            "needle nose", "needle-nose", "long nose", "long-nose"
        ],
        "커터": [
            "커터", "wire cutter", "와이어 커터", "와이어커터"
        ],
        "와이어 스트리퍼": [
            "와이어 스트리퍼", "스트리퍼", "와이어스트리퍼",
            "피복 스트리퍼", "피복제거기", "wire stripper"
        ],
    }

    # DEFAULT_SCENARIO (INFO와 FOCUS 분리)
    DEFAULT_SCENARIO = dedent("""
    [시나리오 지침: 모드 정의와 예시]
    - START: 작업 시작. "시작하자", "준비해"
    - DELIVER: 공구/부품 전달. "니퍼 가져와", "M3 나사 좀"
    - RETURN: 공구 반납. "니퍼 원위치", "도구 정리해"
    - FINISH: 종료/소등. "끝내자", "작업 종료"
    - INFO: 도구에 대한 설명/선택을 묻는 질의. "니퍼 어딨어?" "노란색 도구 뭐야?"
    - FOCUS: 'N번 구역' 조명 비추기 요청. "3번 구역 비춰줘", "1섹션에 불 켜"
    - 규칙: 여러 지시가 있으면 '마지막 명령'을 우선.
    """)

    _KO_BOUND    = r'(?<![가-힣A-Za-z0-9])'
    _KO_BOUND_T  = r'(?![가-힣A-Za-z0-9])'
    _POST_SIMPLE = r'(?:를|을|은|는|이|가|과|와)?'

    _RIPPER_ONLY = re.compile(
        rf'(?<!스트){_KO_BOUND}리퍼{_POST_SIMPLE}{_KO_BOUND_T}'
    )
    # 구역(Section) 추출용
    _SECTION_RE = re.compile(r"(\d+)\s*(?:번|섹션|구역)", re.IGNORECASE)

    class C:
        GREEN, YELLOW, RED, CYAN, END = '\033[92m', '\033[93m', '\033[91m', '\033[96m', '\033[0m'

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

    @staticmethod
    def _last_clause(text: str) -> str:
        """
        여러 문장/명령이 섞여 있을 때 '마지막 명령' 하나만 뽑는다.
        예: "작업대 정리해주고 전선을 구부릴 수 있는 도구도 가져다줘"
            -> "전선을 구부릴 수 있는 도구도 가져다줘"
        """
        if not text:
            return ""
        parts = re.split(r"[\.!?？！\n]", text)
        for p in reversed(parts):
            p = p.strip()
            if p:
                return p
        return text

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

        # RAG 프롬프트 모드 (INFO/FOCUS 포함 6모드)
        mode_template = (
            "당신은 사용자 발화로부터 '모드'만 결정하는 시스템입니다.\n"
            "가능한 모드: START | DELIVER | RETURN | FINISH | INFO | FOCUS\n"
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
    def canonicalize_tool(self, user_text: str) -> Optional[str]:
        txt = self._normalize_kr(user_text)

        hits = []
        # "리퍼" 단독 → 니퍼 계열 처리
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

    def _extract_section(self, user_text: str) -> Optional[int]:
        """ "1번 구역", "3섹션" 등에서 숫자(1-6)를 추출 """
        txt_norm = self._normalize_kr(user_text)
        m = self._SECTION_RE.search(txt_norm)
        if m:
            try:
                sec_num = int(m.group(1))
                if 1 <= sec_num <= 6:
                    return sec_num
            except ValueError:
                pass
        return None

    def _rule_based_mode(self, clause: str) -> Optional[str]:
        """
        아주 명확한 패턴만 처리하는 가벼운 규칙 모드.
        나머지는 전부 RAG/LLM에 맡김.
        입력은 이미 '마지막 문장'이어야 한다.
        """
        t = self._normalize_kr(clause)

        # 질문 느낌 → INFO
        if any(q in t for q in ["어디있", "어디에 있", "어디에있", "뭐야", "무엇이야", "뭔지"]):
            return "INFO"

        # 섹션/구역 언급 → FOCUS 기본 후보
        if self._SECTION_RE.search(t) or any(k in t for k in ["섹션", "구역"]):
            return "FOCUS"

        return None

    # ───── LLM 기반 도구 추출 유틸 ─────
    def llm_extract_tool_and_target(self, user_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        LLM에게 '작업 설명'을 주고, ALLOWED_TOOLS 후보 중 하나를 골라오게 한다.
        출력: (tool_canon, target)
        """
        tool_candidates = ["스탠드", "M3 나사", "니퍼", "커터", "와이어 스트리퍼", "NONE"]

        system_prompt = f"""
        너는 전자/전기 작업 현장의 공구 선택 도우미다.
        아래 한국어 문장을 읽고, 사용자가 사용하려는 공구를 다음 후보 중 하나로 EXACT하게 골라라.

        후보:
        - 스탠드
        - M3 나사
        - 니퍼
        - 커터
        - 와이어 스트리퍼
        - NONE

        도메인:
        - 전선, 회로, 조명, 나사 등을 사용하는 작업 환경이다.
        - 공구 이름이 직접 나오지 않아도, '전선을 구부리다', '전선 피복을 벗기다' 같은 작업 설명을 보고
          어떤 공구가 가장 적합한지 추론해야 한다.
        - 공구에는 색이 지정되어 있다. 색으로만 요청하는 경우에는 색 규칙에 따라 공구를 선택해야 한다.

        색 규칙(색으로만 요청하는 경우):
        - 빨강/빨간/빨간색/빨강색/빨간 도구/빨강 도구 → 니퍼
        - 파랑/파란/파란색/파랑색/파란 도구/파랑 도구 → 커터
        - 노랑/노란/노란색/노랑색/노란 도구/노랑 도구 → 와이어 스트리퍼

        우선순위:
        1) 문장에 공구 이름이 직접 나오면 그 공구를 그대로 선택한다.
        2) 공구 이름은 없고 색만 나오면 위의 색 규칙에 따라 공구를 선택한다.
        3) 공구 이름과 색이 동시에 나오면 공구 이름을 우선한다.
        4) 공구 이름과 색이 둘 다 없고 작업 설명만 나오면, 작업에 가장 적합한 공구를 고른다.
           - 전선을 자르거나 끊는 작업: 커터
           - 전선 피복을 벗기는 작업: 와이어 스트리퍼
           - 전선을 집거나 구부리는 작업: 니퍼
           - 조명을 켜거나 끄는 작업: 스탠드
           - 작은 나사, 볼트 관련 조립/체결: M3 나사
        5) 어떤 공구도 도저히 맞지 않을 때만 "NONE"을 선택한다.
        6) 공구 이외의 잡담, 인사, 설명은 무시하고 작업 의도만 본다.
        7) 출력은 아래 JSON 형식 **한 줄**로만 출력한다.

        예시 1:
        입력: "전선 피복 벗길 수 있는 도구 뭐야?"
        출력: {{"tool": "와이어 스트리퍼", "target": "NONE"}}

        예시 2:
        입력: "전선 구부릴 수 있는 공구 뭐야"
        출력: {{"tool": "니퍼", "target": "NONE"}}

        예시 3:
        입력: "작업대를 정리해줘"
        출력: {{"tool": "NONE", "target": "NONE"}}

        예시 4:
        입력: "스탠드 불 좀 켜줘"
        출력: {{"tool": "스탠드", "target": "NONE"}}

        예시 5:
        입력: "M3 나사 좀 더 필요해"
        출력: {{"tool": "M3 나사", "target": "NONE"}}

        예시 6 (색으로만 요청):
        입력: "빨강 도구 가져다줘"
        출력: {{"tool": "니퍼", "target": "NONE"}}

        예시 7 (색으로만 요청):
        입력: "파란색 공구 좀 줘"
        출력: {{"tool": "커터", "target": "NONE"}}

        예시 8 (색으로만 요청):
        입력: "노란 도구 가져와"
        출력: {{"tool": "와이어 스트리퍼", "target": "NONE"}}

        입력 문장: "{user_text}"
        """

        prompt_to_send = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system_prompt}\n"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )

        try:
            resp = self.llm_model.create_completion(
                prompt=prompt_to_send,
                max_tokens=96,
                temperature=0.0,
                stop=["\n", "<|eot_id|>"],
            )
            raw = resp["choices"][0]["text"].strip()
        except Exception:
            return None, None

        tool_out, target_out = None, None
        try:
            j = json.loads(raw)
            tool_out = j.get("tool")
            target_out = j.get("target")
        except Exception:
            m = re.search(r'"tool"\s*:\s*"([^"]+)"', raw)
            if m:
                tool_out = m.group(1).strip()

        if tool_out not in tool_candidates:
            tool_out = None
        if not target_out:
            target_out = "NONE"

        return tool_out, target_out

    def llm_pick_tool(self, user_text: str) -> Optional[str]:
        """
        1순위: canonicalize_tool 결과
        2순위: LLM 기반 추론(llm_extract_tool_and_target)
        3순위: LLM이 NONE/실패인 경우에만 가벼운 키워드 규칙
        """
        # 1) 텍스트에 직접 나온 공구명/별칭
        explicit = self.canonicalize_tool(user_text)
        if explicit:
            return explicit

        # 2) LLM 추론
        tool_llm, _ = self.llm_extract_tool_and_target(user_text)
        if tool_llm and tool_llm != "NONE":
            return tool_llm

        # 3) LLM도 NONE이면, 아주 얇은 키워드 규칙만 사용
        txt = self._normalize_kr(user_text)

        # 전선 관련
        if "전선" in txt:
            if any(k in txt for k in ["피복", "겉껍질", "껍질", "벗기", "벗겨"]):
                return "와이어 스트리퍼"
            if any(k in txt for k in ["자르", "잘라", "끊", "절단"]):
                return "커터"
            if any(k in txt for k in ["구부리", "굽히", "꺾", "집어", "집어줘", "잡아줘"]):
                return "니퍼"

        # 조명/스탠드 관련
        if any(k in txt for k in ["스탠드", "조명", "작업등", "불 켜", "불켜", "불 꺼", "불꺼"]):
            return "스탠드"

        # 나사/볼트 관련
        if any(k in txt for k in ["나사", "볼트", "m3", "엠쓰리", "엠쓰리 나사"]):
            return "M3 나사"

        # 정말 모르겠으면 None
        return None

    # ───── 모드/명령 파이프라인 (초단순화) ─────
    def parse_and_infer(self, user_text: str) -> Dict[str, Any]:
        """
        사용자 입력을 파싱하여 3가지 키만 반환:
        {"mode": str, "tool_final": str, "section": int}
        - 모드: 마지막 문장만 기준으로 결정
        - 도구: 전체 문장을 기준으로 추론
        """
        mode: Optional[str] = None
        tool: Optional[str] = None
        section: Optional[int] = None

        # 0. 마지막 문장 추출 (모드용)
        last_clause = self._last_clause(user_text)

        # 1. 기본 엔티티 추출 (전체 텍스트 기준)
        explicit_tool = self.canonicalize_tool(user_text)
        section = self._extract_section(user_text)  # (None 또는 1-6)

        # 2. 모드 추론 (가벼운 규칙 → RAG → LLM 순서, 모두 마지막 문장만 사용)
        mode = self._rule_based_mode(last_clause)

        if not mode:
            mode = self.rag_infer_mode(last_clause)

        if not mode:
            system_instructions = (
                "다음 문장은 사용자의 '마지막 명령'만 포함하고 있다.\n"
                "이 문장을 읽고 6모드(START, DELIVER, RETURN, FINISH, INFO, FOCUS) 중 하나로 분류하라.\n"
                "여러 지시가 있었더라도, 이미 정리된 이 한 문장만 보고 판단하면 된다.\n"
                "출력은 영어 대문자 모드명 한 단어."
            )
            prompt_to_send = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"{system_instructions}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
                f"{last_clause}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            try:
                raw = self.llm_model.create_completion(
                    prompt=prompt_to_send,
                    max_tokens=8,
                    temperature=0.0,
                    stop=["\n"],
                )['choices'][0]['text']
                m = self._MODE_RE.search(raw or "")
                mode = m.group(1).upper() if m else None
            except Exception:
                mode = None

        # 3. section이 있으면 무조건 FOCUS로 override
        if section is not None:
            mode = "FOCUS"

        # 4. 도구 결정 로직
        #    - INFO / DELIVER에서는 LLM까지 써서 최대한 찾아보기
        if mode in ("INFO", "DELIVER"):
            tool = explicit_tool or self.llm_pick_tool(user_text)
        else:
            tool = explicit_tool

        if self.verbose:
            print(f"{self.C.CYAN}[단순 결과] mode={mode}, tool={tool}, section={section}{self.C.END}")

        # 5. 요청한 3-key 딕셔너리 반환
        return {
            "mode": mode or "NONE",
            "tool_final": tool or "NONE",
            "section": section if section is not None else 1,
        }

    # ───── ➕ NLG (TTS) 함수 ─────
    @staticmethod
    def _strip_special(s: str) -> str:
        """ TTS 특수 토큰 제거 """
        if not s:
            return s
        s = re.sub(r"<\|[^>]+?\|>", "", s)
        return s.strip()

    def generate_speech(self, task_description: str, style: str = "친절하고 상냥하게") -> str:
        """
        현재 수행할 작업에 대한 자연스러운 TTS 응답을 생성.
        (FOCUS 모드 예시 포함)
        """
        system_prompt = f"""
        당신은 사용자를 돕는 로봇의 친절하고 상냥한 목소리입니다. 사용자의 명령을 확인하고 로봇이 무엇을 할 것인지 자연스럽게 대답해주세요.

        [지침]
        - 문체: {style}
        - 절대로 "알겠습니다" 또는 "네" 라고만 단답형으로 대답하지 마세요.
        - 항상 완전한 문장으로 사용자의 명령을 자연스럽게 복창하며 확인해주세요.

        [좋은 답변의 예시]
        - (작업: START) -> "네, 작업을 시작합니다."
        - (작업: DELIVER, 도구: 커터) -> "네, 커터를 바로 가져다 드릴게요!"
        - (작업: RETURN) -> "작업대를 정리하겠습니다."
        - (작업: FINISH) -> "네, 작업을 종료합니다. 수고하셨습니다!"
        - (작업: INFO, 도구: 니퍼) -> "니퍼를 추천드립니다."
        - (작업: FOCUS, 구역: 3) -> "네, 3번 구역에 조명을 비추겠습니다."
        - (작업: FOCUS, 구역: 1) -> "네, 1번 구역을 비춰드릴게요."

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
