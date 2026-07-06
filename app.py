# Imports
import joblib
import numpy as np
import pandas as pd
import os, gradio as gr
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END


# Weak Topic Model
weakness_model=None
weakness_features = None

def load_weak_model():
    global weakness_model, weakness_features
    try:
        weakness_model = joblib.load('./weakness_model.pkl')
        weakness_features = joblib.load('./features.pkl')
        return 'Model loaded'
    except:
        return 'Please upload the model first'

# Config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=GROQ_API_KEY
)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

vector_db = None
pdf_topics = {}
topic_progress = {}
search_tool = DuckDuckGoSearchRun()

current_quiz = {}
current_quiz_topic = None

# Topic Extraction
import re

def normalize_topic_key(topic: str) -> str:
    topic = (topic or "").strip().lower()
    topic = re.sub(r"\s+", " ", topic)
    topic = topic.strip(" .-•")
    return topic

def extract_candidate_headings(text):

    lines = text.split("\n")
    candidates = []

    for line in lines:

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if (
            4 <= len(line) <= 80
            and not line.endswith("?")
            and not re.match(r"^(page|chapter)\s+\d+", line.lower())
        ):

            title_like = (
                line.istitle()
                or line.isupper()
                or bool(re.match(r"^\d+(\.\d+)*\s+[A-Za-z]", line))
            )

            if title_like:
                candidates.append(line)

    return candidates[:600]


def _dedup_key(topic):

    key = topic.lower().strip()

    if key.endswith("ies") and len(key) > 5:
        key = key[:-3] + "y"
    elif key.endswith("ses") and len(key) > 5:
        key = key[:-2]
    elif key.endswith("s") and not key.endswith("ss") and len(key) > 3:
        key = key[:-1]

    return key


def normalize_topics(topics, preserve_order=False):

    cleaned = []
    seen = {}

    for topic in topics:

        topic = re.sub(r"^\d+(\.\d+)*\.?\s*", "", topic)
        topic = re.sub(r"\(.*?\)", "", topic)
        topic = re.sub(r"\s+", " ", topic)
        topic = topic.strip(" -•.").strip()

        if (
            len(topic) < 4
            or len(topic) > 60
            or len(topic.split()) > 6
        ):
            continue

        lower = topic.lower()

        blocked = [
            "page ",
            "question",
            "assessment",
            "not mentioned",
            "this chapter",
            "describe ",
            "explain ",
            "identify ",
            "configure "
        ]

        if any(x in lower for x in blocked):
            continue

        key = _dedup_key(topic)

        if key in seen:
            existing_idx = seen[key]

            if len(topic) < len(cleaned[existing_idx]):
                cleaned[existing_idx] = topic

            continue

        seen[key] = len(cleaned)
        cleaned.append(topic)

    if preserve_order:
        return cleaned

    return sorted(cleaned)

def extract_topics(text):

    text = (text or "").strip()

    if not text:
        return []

    candidates = extract_candidate_headings(text)

    prompt = f"""
Extract STUDY TOPICS from the educational material below.

This text was extracted from a PDF and may have LOST its original
formatting (bold headings, font sizes, slide titles, etc. may not be
visible anymore). Some material may be a textbook with plain running
paragraphs, some may be slides with short bullet points. Either way,
read the actual content and identify the underlying concepts being
taught -- do NOT rely on which lines "look like" headings.

Rules:
- Return topic names only, one per line
- Each topic should be a short noun phrase (max 5 words)
- Base topics on concepts actually discussed in the text
- No explanations, no complete sentences, no questions
- Merge duplicates or near-duplicate topics
- Ignore commands, examples, page numbers, IDs, and quiz/question text

Material:
{text[:6000]}
"""

    try:
        result = llm.invoke(prompt).content
    except Exception:
        result = ""

    raw = []

    for line in result.splitlines():
        line = line.strip()

        if line:
            raw.append(line)

    topics = normalize_topics(raw, preserve_order=True)

    if len(topics) < 8:
        fallback = normalize_topics(candidates, preserve_order=True)

        if len(fallback) > len(topics):
            topics = fallback

    return topics

# PDF Processing
def chunk_text(text, chunk_size=3000, max_chunks=6):

    text = text or ""

    chunks = [
        text[i:i + chunk_size]
        for i in range(0, len(text), chunk_size)
        if text[i:i + chunk_size].strip()
    ]

    if not chunks:
        return []

    if len(chunks) <= max_chunks:
        return chunks

    step = len(chunks) / max_chunks
    return [chunks[int(i * step)] for i in range(max_chunks)]


def process_pdfs(files):
    global vector_db, pdf_topics, topic_progress

    if not files:
        return "No PDFs uploaded. Please upload at least one PDF."

    from concurrent.futures import ThreadPoolExecutor, as_completed

    docs = []
    pdf_topics = {}

    for f in files:
        loader = PyPDFLoader(f.name)
        pages = loader.load()
        docs.extend(pages)

        full_text = "\n".join(p.page_content for p in pages)
        text_chunks = chunk_text(full_text)

        chunk_topics = [None] * len(text_chunks)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(extract_topics, chunk): idx
                for idx, chunk in enumerate(text_chunks)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunk_topics[idx] = future.result()
                except Exception:
                    chunk_topics[idx] = []

        all_topics = [t for topics in chunk_topics for t in (topics or [])]

        topics = normalize_topics(all_topics, preserve_order=True)

        pdf_topics[os.path.basename(f.name)] = topics

        for topic in topics:
            key = normalize_topic_key(topic)
            topic_progress.setdefault(key, {'covered': False, 'score': 0, 'attempts': 0, 'best_score': 0})

    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="/var/data/EPS_db"
    )

    summary = ", ".join(f"{name} ({len(t)} topics)" for name, t in pdf_topics.items())
    return f"Processing complete. {summary}"

# Retrieval with Relevance Routing
def retrieve_context(topic, k=3):

    if vector_db is None:
        return ""

    docs = vector_db.similarity_search(
        topic,
        k=k
    )

    if not docs:
        return ""

    return "\n\n".join(
        d.page_content for d in docs
    )


def route_query(topic):

    if vector_db is None:
        return "web"

    results = vector_db.similarity_search_with_score(
        topic,
        k=3
    )

    if not results:
        return "web"

    best_doc, distance = results[0]

    print(f"Routing score: {distance:.4f}")

    if distance < 0.80:
        return "pdf"

    if distance < 1.20:
        return "both"

    return "web"

# Agents Functions
import re
import json

def generate_notes(topic):

    raw_topic = topic

    if not raw_topic:
        return 'Please select a topic.'

    context = retrieve_context(raw_topic)

    if not context.strip():
        return f'Topic not found in uploaded PDFs: {raw_topic}'

    prompt = f'''Use ONLY PDF content.

Topic: {raw_topic}

Context:
{context}

Create:
1. Definition
2. Key Concepts
3. Examples
4. Exam Tips

Rules:
- Do NOT use a Markdown table (no | pipe-separated rows) anywhere in
  the answer, even for comparisons or term/definition pairs. Use
  headings, bold key terms, and bullet/numbered lists instead -- this
  reads better on mobile screens than a table does.
'''

    answer = llm.invoke(prompt).content

    key = normalize_topic_key(raw_topic)

    topic_progress.setdefault(
        key,
        {
            "covered": False,
            "score": 0,
            "attempts": 0,
            "best_score": 0
        }
    )

    topic_progress[key]["covered"] = True

    return answer

def web_research(topic):
    try:
        return search_tool.run(topic)
    except Exception as e:
        return str(e)


def normalize_quiz(quiz):

    for q in quiz.get("questions", []):

        cleaned = []

        for opt in q.get("options", []):

            opt = re.sub(r'^[A-D][.)]\s*', '', str(opt))
            opt = re.sub(r'^\([A-D]\)\s*', '', opt)

            cleaned.append(opt.strip())

        q["options"] = cleaned

    return quiz


def generate_quiz(topic):

    global current_quiz, current_quiz_topic

    context = retrieve_context(topic)

    prompt = f'''
    Using ONLY the supplied PDF context, create exactly 5 MCQs.

    Return ONLY valid JSON in this format:

    {{
      "questions":[
        {{
          "question":"...",
          "options":["A","B","C","D"],
          "answer":"A",
          "explanation":"Why the answer is correct"
        }}
      ]
    }}

    Context:
    {context[:3000]}
    '''

    response = llm.invoke(prompt).content

    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        quiz_json = response[start:end]

        current_quiz = normalize_quiz(json.loads(quiz_json))

        current_quiz_topic = topic

        formatted = ""

        for i, q in enumerate(current_quiz["questions"], 1):
            formatted += f"Q{i}. {q['question']}\n"

            for idx, opt in enumerate(q["options"]):
                letter = chr(65 + idx)
                formatted += f"{letter}. {opt}\n"

            formatted += "\n"

        return formatted

    except Exception:
        current_quiz = {}
        current_quiz_topic = None
        return response


def score_quiz(topic, q1, q2, q3, q4, q5):

    global current_quiz, current_quiz_topic

    if not current_quiz or not current_quiz_topic:
        return "Generate a quiz first."

    key = normalize_topic_key(current_quiz_topic)

    answers = [q1, q2, q3, q4, q5]

    correct = 0

    incorrect_feedback = []
    correct_feedback = []

    for i, (question, user_answer) in enumerate(
        zip(current_quiz["questions"], answers),
        1
    ):

        correct_answer = str(
            question["answer"]
        ).strip()

        explanation = question.get(
            "explanation",
            "Review the topic notes for this concept."
        )

        question_text = question["question"]

        user_answer = (
            str(user_answer).strip()
            if user_answer
            else "-"
        )

        if user_answer.upper() == correct_answer.upper():

            correct += 1

            correct_feedback.append(
f"""
### ✅ Q{i}: Correct

Question: {question_text}

Your Answer: {user_answer}

Explanation: {explanation}

---
"""
            )

        else:

            incorrect_feedback.append(
f"""
### ❌ Q{i}: Incorrect

Question: {question_text}

Your Answer: {user_answer}

Correct Answer: {correct_answer}

Explanation: {explanation}

---
"""
            )

    total = len(current_quiz["questions"])

    score = round(
        (correct / total) * 100,
        2
    )

    # ✅ Update dashboard progress
    topic_progress.setdefault(

        key,

        {
            "covered": True,
            "score": 0,
            "attempts": 0,
            "best_score": 0
        }

    )

    topic_progress[key]["covered"] = True

    topic_progress[key]["score"] = score

    topic_progress[key]["attempts"] += 1

    topic_progress[key]["best_score"] = max(

        topic_progress[key]["best_score"],

        score

    )

    incorrect_text = (
        "\n".join(incorrect_feedback)
        if incorrect_feedback
        else "None 🎉"
    )

    correct_text = (
        "\n".join(correct_feedback)
        if correct_feedback
        else "None"
    )

    return f"""
# 📊 Quiz Results

Score: {score}%
Correct: {correct}/{total}
Attempts: {topic_progress[key]['attempts']}
Best Score: {topic_progress[key]['best_score']}%

---

## ❌ Questions To Review

{incorrect_text}

---

## ✅ Correct Answers & Explanations

{correct_text}
"""

def recommendations():

    display_map = get_topic_display_map()

    weak = []
    medium = []
    strong = []

    for key, display_name in display_map.items():

        data = topic_progress.get(key, {})

        if data.get("attempts", 0) == 0:
            weak.append(display_name)
            continue

        score = data.get("score", 0)

        if score < 60:
            weak.append(display_name)
        elif score < 75:
            medium.append(display_name)
        else:
            strong.append(display_name)

    output = ""

    if weak:
        output += "🔴 Immediate Revision Needed\n"
        output += "\n".join(sorted(weak))
        output += "\n\n"

    if medium:
        output += "🟡 Practice More\n"
        output += "\n".join(sorted(medium))
        output += "\n\n"

    if strong:
        output += "🟢 Well Understood\n"
        output += "\n".join(sorted(strong))
        output += "\n\n"

    return output.strip() if output else "No topics found. Process a PDF first."

# LangGraph Multi-Agent Workflow
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    topic: str
    route: str

    pdf_result: Optional[str]
    web_result: Optional[str]

    synthesized: Optional[str]

    final_answer: Optional[str]
    judge_reason: Optional[str]

def router_agent(state):

    route = route_query(
        state["topic"]
    )

    print("Selected route:", route)

    return {
        "route": route
    }

def pdf_agent(state):

    raw_topic = state["topic"]
    key = normalize_topic_key(raw_topic)

    result = generate_notes(raw_topic) # keep raw for retrieval quality

    topic_progress.setdefault(
        key,
        {"covered": False, "score": 0, "attempts": 0, "best_score": 0}
    )

    topic_progress[key]["covered"] = True

    return {
        "pdf_result": result
    }

def web_agent(state):

    topic = state["topic"]

    result = web_research(topic)

    return {
        "web_result": result
    }

def synth_agent(state):

    route = state.get("route", "")

    source_labels = {
        "pdf":  "📄 Source: PDF",
        "web":  "🌐 Source: Web",
        "both": "📄🌐 Source: PDF + Web",
    }
    source_badge = source_labels.get(route, "")

    prompt = f"""
You are a synthesis agent.

Create the final exam answer.

PDF information:
{state.get("pdf_result", "")}

Web information:
{state.get("web_result", "")}

IMPORTANT:
Use Markdown formatting.

Output EXACTLY:

1. Definition
- Give a clear detailed definition.

2. Key Concepts
- List important terms.
- Explain each term.
- Include differences where relevant.

3. Examples
- Provide multiple practical examples.
- Group examples if possible.

4. Exam Tips
- Give revision points.
- Include common exam mistakes.


Rules:
- Do NOT summarize too aggressively.
- Keep important details.
- Prefer PDF content.
- Do not remove concepts.
- Do NOT use a Markdown table (no | pipe-separated rows) anywhere in
  the answer, even for comparisons or term/definition pairs. Use
  headings, bold key terms, and bullet/numbered lists instead -- this
  reads better on mobile screens than a table does.
"""

    answer = llm.invoke(prompt).content

    if source_badge:
        answer = f"{source_badge}\n\n---\n\n{answer}"

    return {
        "synthesized": answer
    }

def critic_agent(state):

    return {
        "judge_reason": "",
        "final_answer": state.get("synthesized", "")
    }

builder = StateGraph(AgentState)

builder.add_node(
    "router",
    router_agent
)

builder.add_node(
    "pdf_agent",
    pdf_agent
)

builder.add_node(
    "web_agent",
    web_agent
)

builder.add_node(
    "synth_agent",
    synth_agent
)

builder.add_node(
    "critic_agent",
    critic_agent
)

builder.set_entry_point(
    "router"
)

builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "pdf": "pdf_agent",
        "web": "web_agent",
        "both": "pdf_agent"
    }
)

builder.add_edge(
    "pdf_agent",
    "synth_agent"
)

builder.add_edge(
    "web_agent",
    "synth_agent"
)

builder.add_edge(
    "synth_agent",
    "critic_agent"
)

builder.add_edge(
    "critic_agent",
    END
)


workflow = builder.compile()

# Dashboard Helpers
def dashboard():
    text = ''
    for pdf, topics in pdf_topics.items():
        text += f'\n📘 {pdf}\n\n'
        covered = 0

        for topic in topics:

            key = normalize_topic_key(topic)

            status = topic_progress.get(normalize_topic_key(topic), {}).get('covered', False)

            text += ('✅ ' if status else '❌ ') + topic + '\n'
            covered += int(status)

        pct = round((covered/max(len(topics),1))*100,1)
        text += f'\nProgress: {pct}%\n'
        text += '-'*40 + '\n'

    return text

def get_topic_display_map():

    seen = {}

    for topics in pdf_topics.values():
        for t in topics:
            key = normalize_topic_key(t)
            if key not in seen:
                seen[key] = t

    return seen


def get_topic_choices():
    return sorted(get_topic_display_map().values())

def ask_topic_trace(topic):
    events = workflow.stream({
        "topic": topic,
        "route": ""
    })

    trace = []
    for event in events:
        trace.append(json.dumps(event, indent=2))

    return "\n\n---\n\n".join(trace)

def detable(text):

    import re

    lines = text.split("\n")
    output = []
    headers = []
    in_table = False

    for line in lines:

        if re.match(r"\s*\|", line) and "|" in line:

            cells = [c.strip() for c in line.strip().strip("|").split("|")]

            if all(re.match(r"[-: ]+$", c) for c in cells if c):
                continue

            if not in_table:
                headers = cells
                in_table = True
            else:
                pairs = []
                for h, v in zip(headers, cells):
                    if h and v:
                        pairs.append(f"**{h}:** {v}")
                if pairs:
                    output.append("\n".join(pairs))
                    output.append("\n---")

        else:
            in_table = False
            headers = []
            output.append(line)

    return "\n".join(output)


def ask_topic_with_progress(topic):

    yield "⏳ Generating notes, please wait..."
    yield ask_topic(topic)

def ask_topic(topic):

    if not topic:
        return "Please select a topic."

    key = normalize_topic_key(topic)

    result = workflow.invoke({

        "topic": topic,
        "route": "",

        "pdf_result": None,
        "web_result": None,

        "synthesized": None,

        "final_answer": None,
        "judge_reason": None

    })

    topic_progress.setdefault(

        key,

        {
            "covered": False,
            "score": 0,
            "attempts": 0,
            "best_score": 0
        }

    )

    topic_progress[key]["covered"] = True

    return detable(result["final_answer"])


# Dataset Dashboard
import pandas as pd

def get_dataset_df():

    rows = []

    for topic, data in topic_progress.items():

        rows.append({
            "topic": topic,
            "score": data.get("score", 0),
            "attempts": data.get("attempts", 0),
            "best_score": data.get("best_score", 0),
            "covered": data.get("covered", False)
        })

    df = pd.DataFrame(rows)

    if len(df) == 0:
        df = pd.DataFrame(
            columns=["topic","score","attempts","best_score","covered"]
        )

    return df.sort_values("topic").reset_index(drop=True)

def refresh_dataset():
    return get_dataset_df()

def download_dataset_csv():
    df = get_dataset_df()
    csv_path = "topic_progress_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# Weak Topic Dashboard
def build_topic_feature_df():

    rows=[]

    for topic,data in topic_progress.items():

        score=float(data.get("score",0))
        attempts=float(data.get("attempts",0))
        best=float(data.get("best_score",0))
        covered=int(bool(data.get("covered",False)))

        efficiency=score/(attempts+1)
        attempt_pressure=attempts*(100-score)
        recent_performance=best/100
        improvement=best-score
        gap_to_perfection=100-best

        rows.append({
            "topic":topic,
            "score":score,
            "attempts":attempts,
            "best_score":best,
            "covered":covered,
            "efficiency":efficiency,
            "attempt_pressure":attempt_pressure,
            "recent_performance":recent_performance,
            "improvement":improvement,
            "gap_to_perfection":gap_to_perfection
        })

    return pd.DataFrame(rows)

def get_status(row):

    if row["attempts"] == 0:
        return "⚪ Not Attempted"

    if row["best_score"] < 40:
        return "🔴 Weak"

    if row["best_score"] < 75:
        return "🟡 Improving"

    return "🟢 Strong"


def predict_weak_topics():

    if weakness_model is None:
        return pd.DataFrame({"Status":["Load weakness_model.pkl first"]})

    df=build_topic_feature_df()

    if len(df)==0:
        return pd.DataFrame()

    feature_cols = weakness_features

    probs=[]

    for _,row in df.iterrows():

        if (
            row["score"]==0 and
            row["attempts"]==0 and
            row["best_score"]==0
        ):
            probs.append(None)

        else:

            row_df=pd.DataFrame([{
                col:row[col]
                for col in feature_cols
            }])

            p=float(
                weakness_model.predict_proba(row_df)[0][1]
            )

            probs.append(round(p*100,1))

    out=df.copy()

    out["weak_probability"]=probs

    out["prediction"]=np.where(
        out["weak_probability"].isna(),
        "⚪ Not Attempted",
        np.where(
            out["weak_probability"]>=50,
            "🔴 Weak",
            "🟢 Normal"
        )
    )

    return out.sort_values(
        by=["weak_probability"],
        ascending=False,
        na_position="last"
    )[
        [
            "topic",
            "prediction",
            "weak_probability",
            "score",
            "attempts",
            "best_score"
        ]
    ]

def refresh_weak_dashboard():
    return predict_weak_topics()

# Model Test Dashboard
def test_ml_risk():

    df = build_topic_feature_df()

    if len(df) == 0:
        return "No topics available"

    valid = df[
        ~(
            (df["score"] == 0) &
            (df["attempts"] == 0) &
            (df["best_score"] == 0)
        )
    ].copy()

    if len(valid) == 0:
        return "No attempted topics yet"

    feature_cols = weakness_features

    valid["weakness_probability"] = (
        weakness_model
        .predict_proba(valid[feature_cols])[:, 1] * 100
    ).round(1)

    highest = (
        valid
        .sort_values(
            "weakness_probability",
            ascending=False
        )
        .head(5)
    )

    lowest = (
        valid
        .sort_values(
            "weakness_probability",
            ascending=True
        )
        .head(5)
    )

    corr = "N/A"

    if valid["score"].nunique() > 1:
        corr = (
            f"{valid['weakness_probability'].corr(valid['score']):.3f}"
        )

    bucket = (
       valid.assign(
         weakness_level=pd.cut(
            valid["weakness_probability"],
            [0, 40, 70, 100],
            labels=[
                "Low Weakness",
                "Medium Weakness",
                "High Weakness"
            ]
         )
       )
       .groupby("weakness_level", observed=True)["score"]
       .mean()
       .round(1)
       .to_string()
    )

    highest_table = (
        highest[
            [
                "topic",
                "weakness_probability",
                "score"
            ]
        ]
        .to_markdown(index=False)
    )


    lowest_table = (
        lowest[
            [
                "topic",
                "weakness_probability",
                "score"
            ]
        ]
        .to_markdown(index=False)
    )


    return f"""
# 📊 ML Weak Topic Detection Report


## 🔥 Topics Most Likely to Need Revision

{highest_table}


---

## 🟢 Topics Performing Well

{lowest_table}


---

## 📈 Correlation (Weakness Probability vs Score)

{corr}


---

## 🎯 Average Score by Predicted Weakness Level

```text
{bucket}
"""

# Custom CSS styling for the Gradio interface
CUSTOM_CSS = '''
@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap");

* {
    font-family: "Inter", sans-serif !important;
}

.gradio-container {
    width: 600px !important;
    max-width: 100% !important;
    margin: auto !important;
}

@media (max-width: 800px) {
    .gradio-container {
        width: 100% !important;
        margin: auto !important;
    }
}

h1,h2,h3 {
    font-weight: 700 !important;
}

.gr-button {
    border-radius: 12px !important;
}

textarea,input,select {
    font-size: 17px !important;
}

footer {
    display:none !important;
}

#notes_output .prose {
    max-width: 500px;
    font-size: 17px;
    margin: 0 auto;
}

#notes_output .progress-bar,
#notes_output .loader,
#notes_output .generating {
    display: none !important;
}

.upload-tab {
    max-width: 500px;
    margin: 0 auto;   /* centers it */
}

#scores_output .prose {
    max-width: 500px;
    font-size: 17px;
    margin: 0 auto;   /* centers it */
}
'''

# Gradio UI
theme = gr.themes.Soft(
    spacing_size='lg',
    radius_size='lg'
)

with gr.Blocks(theme=theme, css=CUSTOM_CSS) as demo:

    gr.Markdown('# 🎓 AI Exam Preparation System')

    with gr.Tab('Upload PDFs'):
        files = gr.File(file_count='multiple')
        process_btn = gr.Button('Process PDFs')
        process_out = gr.Textbox(
            label="",
            show_label=False,
            lines=5
        )
        process_btn.click(process_pdfs, files, process_out)

    with gr.Tab('Learning Dashboard'):
        dash_btn = gr.Button('Refresh Dashboard')
        dash_out = gr.Textbox(
            label="",
            show_label=False,
            lines=20
        )
        dash_btn.click(dashboard, outputs=dash_out)

    with gr.Tab('Study Notes'):
        topic = gr.Dropdown(choices=[], label='Topic')
        load_topics = gr.Button('Load Topics')
        load_topics.click(lambda: gr.update(choices=get_topic_choices()), outputs=topic)

        notes_btn = gr.Button('Generate Notes')
        notes_out = gr.Markdown(elem_id="notes_output")
        notes_btn.click(
            ask_topic_with_progress,
            inputs=topic,
            outputs=notes_out
        )

    with gr.Tab('Quiz'):
        quiz_topic = gr.Dropdown(choices=[], label='Topic')
        load_quiz_topics = gr.Button('Load Topics')
        load_quiz_topics.click(lambda: gr.update(choices=get_topic_choices()), outputs=quiz_topic)

        quiz_btn = gr.Button('Generate Quiz')
        quiz_display = gr.Textbox(
            label="",
            show_label=False,
            lines=18
        )
        gr.Markdown('### Submit Answers (A/B/C/D)')

        q1 = gr.Radio(['A','B','C','D'], label='Question 1')
        q2 = gr.Radio(['A','B','C','D'], label='Question 2')
        q3 = gr.Radio(['A','B','C','D'], label='Question 3')
        q4 = gr.Radio(['A','B','C','D'], label='Question 4')
        q5 = gr.Radio(['A','B','C','D'], label='Question 5')

        quiz_btn.click(
            generate_quiz, quiz_topic, quiz_display
        ).then(
            lambda: (None, None, None, None, None),
            outputs=[q1, q2, q3, q4, q5]
        )

        submit_btn = gr.Button('Submit Quiz')

        score_output = gr.Markdown(elem_id="scores_output")

        submit_btn.click(
            score_quiz,
            inputs=[quiz_topic, q1, q2, q3, q4, q5],
            outputs=score_output
        )

    with gr.Tab('Recommendations'):
        rec_btn = gr.Button('Get Recommendations')
        rec_out = gr.Textbox(
            label="",
            show_label=False,
            lines=10
        )
        rec_btn.click(recommendations, outputs=rec_out)

    with gr.Tab("Weak Topic Dashboard"):

        gr.Markdown("## 🧠 Weak Topic Detection")

        model_btn=gr.Button("Load Weakness Model")
        model_status=gr.Textbox(
            label="",
            show_label=False
        )

        model_btn.click(
            load_weak_model,
            outputs=model_status
        )

        weak_btn=gr.Button("Analyze Topics")

        weak_table=gr.Dataframe(
            interactive=False
        )

        weak_btn.click(
            refresh_weak_dashboard,
            outputs=weak_table
        )

    with gr.Tab("ML Risk Test Dashboard"):
        test_btn = gr.Button("Run ML Evaluation")
        test_out = gr.Markdown()

        test_btn.click(
            test_ml_risk,
            outputs=test_out
        )

    with gr.Tab("Dataset Dashboard"):
        gr.Markdown("## 📊 Dataset Dashboard")
        dataset_table = gr.Dataframe(
        value=get_dataset_df(),
        interactive=False,
        wrap=True
        )

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh Dataset")
            download_btn = gr.Button("⬇️ Download CSV")

        csv_file = gr.File(label="Dataset CSV")

        refresh_btn.click(
            fn=refresh_dataset,
            outputs=dataset_table
        )

        download_btn.click(
            fn=download_dataset_csv,
            outputs=csv_file
        )

    with gr.Tab("Agent Trace"):
        trace_topic = gr.Dropdown(choices=[], label="Topic")

        load_trace_btn = gr.Button("Load Topics")
        run_trace_btn = gr.Button("Run Trace")

        trace = gr.Textbox(label="Agent Trace", lines=25)

        load_trace_btn.click(
            lambda: gr.update(choices=get_topic_choices()),
            outputs=trace_topic
        )

        run_trace_btn.click(
            ask_topic_trace,
            inputs=trace_topic,
            outputs=trace
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        debug=False,
        share=False
    )