"""Local EDA workspace: chat, data, live traces and inspectable memory."""
import json
import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from agent import Agent, CloudGigaChat

load_dotenv(Path(__file__).with_name(".env"))
st.set_page_config(page_title="EDA с памятью", layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container {padding: 1.3rem 1.5rem 1rem; max-width: 100%;}
h1 {font-size: 1.65rem; padding-top: 0;}
h3 {font-size: 1.1rem;}
[data-testid="stFileUploaderDropzone"] {padding: .35rem .5rem; min-height: 0;}
[data-testid="stFileUploaderDropzone"] small {display: none;}
[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] {display: none;}
[data-testid="stChatMessage"] {padding: .65rem;}
button {white-space: normal;}
[data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {white-space: pre-wrap !important; overflow-wrap: anywhere;}
</style>""", unsafe_allow_html=True)

if st.session_state.get("ui_version") != 4:
    llm = None
    if os.getenv("GIGACHAT_CLOUD_KEY") and os.getenv("EDA_OFFLINE") != "1":
        llm = CloudGigaChat(os.environ["GIGACHAT_CLOUD_KEY"])
    st.session_state.agent = Agent(os.getenv("EDA_DATA_DIR", "data"), llm=llm, embeddings=llm)
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.chat = []
    st.session_state.pending = None
    st.session_state.output_current = st.session_state.agent.has_current_result()
    st.session_state.ui_version = 4
agent = st.session_state.agent
# A renderer belongs to one script run; discard the previous run's callback.
agent.on_event = None


def load_tables(tables, label, name):
    """Replace raw tables only after all input files have parsed successfully."""
    if not tables:
        raise ValueError("Сначала выберите хотя бы один CSV")
    if any(not name or not len(df.columns) for name, df in tables.items()):
        raise ValueError("У каждой таблицы должно быть имя и колонки")
    agent.load_dataset(name, tables)
    st.session_state.active_version_picker = agent.dataset_version["name"]
    st.session_state.output_current = False
    agent.emit("python.info", message="Загружена поставка", source=label, rows={k: len(v) for k,v in tables.items()})
    st.session_state.notice = "Сохранена версия «" + agent.dataset_version["name"] + "»: " + ", ".join(f"{k} ({len(v)} строк)" for k,v in tables.items())


def queue_message():
    text = st.session_state.get("prompt", "").strip()
    if text and not st.session_state.pending:
        st.session_state.chat.append(("user", text))
        st.session_state.pending = text


def render_tokens():
    state = agent.graph.get_state({"configurable": {"thread_id": st.session_state.thread}}).values
    with token_panel.container():
        context, input_count, output_count, calls = st.columns(4)
        context.metric("Контекст · оценка", f"≈ {agent.context_tokens(state):,} / {agent.context_limit:,}".replace(",", " "))
        input_count.metric("Вход API · факт", f"{agent.api_totals['input_tokens']:,}".replace(",", " "))
        output_count.metric("Выход API · факт", f"{agent.api_totals['output_tokens']:,}".replace(",", " "))
        calls.metric("Вызовы GigaChat", agent.api_totals["calls"])
        st.caption("Контекст: рабочая история, резюме и служебные данные. Расход API — по usage всех ответов, включая суммаризацию и повторные запросы, за время работы приложения.")
        if agent.context_notice:
            if agent.context_notice.startswith("Запускаю"):
                st.info(agent.context_notice)
            else:
                st.caption(agent.context_notice)


st.title("EDA с памятью")
st.caption(("GigaChat · Cloud.ru" if agent.llm else "Локальный режим · /remember JSON, /search текст, /run дата") + " · CSV: имя файла = имя таблицы")
token_panel = st.empty()
render_tokens()
def open_upload():
    st.session_state.upload_open = True
    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
    st.session_state.new_version_name = ""


load, new_chat, reset = st.columns([3, 1, 1])
with load:
    st.button("Загрузить новую версию", on_click=open_upload, disabled=st.session_state.get("upload_open", False), use_container_width=True)
with new_chat:
    if st.button("Новый чат", use_container_width=True, help="Новая сессия. Таблицы и правила сохраняются."):
        st.session_state.thread = str(uuid.uuid4())
        st.session_state.chat = []
        st.session_state.pending = None
        st.session_state.notice = "Начат новый чат. Правила и таблицы сохранены."
with reset:
    if st.button("С чистого листа", use_container_width=True, help="Удалить правила, чат, все версии данных, результаты и журнал."):
        agent.forget()
        st.session_state.upload_open = False
        st.session_state.chat = []
        st.session_state.pending = None
        st.session_state.output_current = False
        st.session_state.thread = str(uuid.uuid4())
        st.session_state.notice = "Память, таблицы и результаты очищены."
if st.session_state.get("upload_open", False):
    generation = st.session_state.upload_generation
    with st.form(f"upload_version_{generation}"):
        st.subheader("Новая версия данных")
        st.caption("Укажите название и файлы в одной форме. После сохранения новые таблицы полностью заменят текущие на экране; предыдущая версия останется в архиве.")
        name = st.text_input("Название версии", key="new_version_name", placeholder="Например, портфель за июль 2026")
        source = st.radio("Данные", ["Мои CSV-файлы", "Пример: июнь (batch_1)", "Пример: июль (batch_2)"], key=f"upload_source_{generation}", horizontal=True)
        files = st.file_uploader("CSV-файлы новой версии", type="csv", accept_multiple_files=True, key=f"uploads_{generation}")
        st.caption("Имя CSV — имя таблицы. Выберите все таблицы новой выгрузки; старые файлы к ней не добавляются.")
        save, cancel = st.columns([2, 1])
        with save:
            submitted = st.form_submit_button("Сохранить и использовать", type="primary", use_container_width=True)
        with cancel:
            cancelled = st.form_submit_button("Отмена", use_container_width=True)
        if cancelled:
            st.session_state.upload_open = False
            st.rerun()
        if submitted:
            try:
                if not name.strip():
                    raise ValueError("Укажите название версии")
                if source == "Мои CSV-файлы":
                    if not files:
                        raise ValueError("Выберите CSV-файлы новой версии")
                    if len({Path(f.name).stem for f in files}) != len(files):
                        raise ValueError("Имена таблиц не должны повторяться")
                    tables = {Path(f.name).stem: pd.read_csv(f) for f in files}
                else:
                    if files:
                        raise ValueError("Для загруженных файлов выберите «Мои CSV-файлы» или уберите файлы, чтобы использовать пример")
                    batch = "batch_1" if "batch_1" in source else "batch_2"
                    folder = Path(__file__).parent / "examples" / "credit_portfolio" / batch / "raw"
                    tables = {p.stem: pd.read_csv(p) for p in sorted(folder.glob("*.csv"))}
                load_tables(tables, source, name)
                st.session_state.upload_open = False
                st.rerun()
            except Exception as exc:
                agent.emit("python.error", message="Ошибка загрузки версии", error=str(exc))
                st.error(str(exc))


def select_version():
    agent.use_dataset(st.session_state.active_version_picker)
    st.session_state.output_current = agent.has_current_result()
    st.session_state.notice = "Выбрана версия «" + agent.dataset_version["name"] + "». /run будет работать с ней."


versions = agent.list_datasets()
if versions:
    names = [v["name"] for v in versions]
    if st.session_state.get("active_version_picker") not in names:
        st.session_state.active_version_picker = agent.dataset_version["name"] if agent.dataset_version else names[0]
    st.selectbox("Версия данных для /run", names, key="active_version_picker", on_change=select_version)
    if agent.dataset_version:
        st.caption("Выбрана версия: " + agent.dataset_version["name"] + " · создана " + agent.dataset_version["created_at"] + " · сохранена в SQLite")
else:
    st.caption("Нажмите «Загрузить новую версию»: в одной форме задайте название и выберите CSV или пример.")
if notice := st.session_state.get("notice"):
    st.caption(notice)

# Diagnostics stay in the sidebar; details update during synchronous API calls.
with st.sidebar:
    st.subheader("Журнал выполнения")
    st.caption("Последние 200 событий этой сессии приложения. Запросы и ответы могут содержать данные таблиц.")
    if st.button("Очистить журнал"):
        agent.events.clear()
    llm_tab, tools_tab, python_tab = st.tabs(["GigaChat", "Инструменты", "Python"])
    log_panels = []
    for tab in (llm_tab, tools_tab, python_tab):
        with tab:
            log_panels.append(st.empty())

chat_col, data_col, memory_col = st.columns([1.1, 1.25, 1.05], gap="medium")
with chat_col:
    st.subheader("Диалог")
    chat_box = st.container(height=580, border=True)
    with chat_box:
        if not st.session_state.chat:
            st.caption("Загрузите таблицы и опишите их. Например: «Запомни: application_id — ключ таблицы applications».")
        for role, text in st.session_state.chat:
            with st.chat_message(role):
                st.write(text)
    st.chat_input("Описание, правило или вопрос…", key="prompt", on_submit=queue_message, disabled=bool(st.session_state.pending))
    with st.expander("Как попробовать"):
        st.write("1. Нажмите «Загрузить новую версию», задайте название и выберите июньский пример. 2. Сохраните правила в чате. 3. Выполните /run 2026-06-30. 4. Через ту же кнопку загрузите новую версию с июльским примером и выполните /run 2026-07-31.")
        commands = Path(__file__).parent / "examples" / "credit_portfolio" / "commands.txt"
        st.download_button("Скачать команды примера", commands.read_bytes(), file_name="commands.txt")
        st.caption("Отправляйте первые пять строк по одной. «Новый чат» сохраняет правила; «С чистого листа» удаляет их.")

with data_col:
    st.subheader("Таблицы и результаты")
    data_panel = st.empty()
with memory_col:
    st.subheader("Текущая память")
    memory_panel = st.empty()


def render_logs():
    groups = [
        [e for e in agent.events if e["kind"].startswith(("llm.", "embedding.", "context."))],
        [e for e in agent.events if e["kind"].startswith(("tool.", "transform.", "retrieval.", "agent.", "rule."))],
        [e for e in agent.events if e["kind"].startswith("python.")],
    ]
    for panel, events in zip(log_panels, groups):
        with panel.container():
            if not events:
                st.caption("Событий пока нет")
            for event in reversed(events[-25:]):
                label = event["time"][11:19] + " · " + event["kind"] + " " + event.get("name", "")
                with st.expander(label, expanded=event is events[-1]):
                    st.json(event)


def render_memory():
    with memory_panel.container():
        records = agent.memory_snapshot()
        st.caption(f"{len(records)} записей · дата выбора: {agent.last_as_of or 'сегодня'}")
        with st.container(height=650, border=True):
            if not records:
                st.info("Память пуста. Сохраните описание таблицы или правило в чате.")
            for r in records:
                state = {"active":"активна", "future":"будущая", "superseded":"заменена"}[r["state"]]
                kind = {"procedural":"Правило обработки", "semantic":"Описание / факт"}[r["kind"]]
                st.markdown(f"**{r['id']} · v{r['version']}** · {state}")
                st.caption(kind + " · действует с " + r["valid_from"])
                st.write(r["description"])
                st.caption("Хранение: " + r["retention"])
                st.caption("Создана: " + (r.get("created_at") or "нет данных (старая запись)") + " · источник: " + r.get("source", "user"))
                if r.get("source_text") and r["source_text"] != r["description"]:
                    with st.expander("Исходная реплика"):
                        st.write(r["source_text"])
                if r["steps"]:
                    with st.expander("Шаги обработки"):
                        st.json(r["steps"])
                st.divider()
            st.caption("Диалог и слоты: RAM, до перезапуска процесса. Правила: SQLite, между сессиями. Автоматического TTL пока нет.")
            state = agent.graph.get_state({"configurable": {"thread_id": st.session_state.thread}}).values
            with st.expander("Слоты и резюме текущего чата"):
                st.json(state.get("slots", {}))
                st.write(state.get("summary") or "Резюме пока не создавалось")


def render_data():
    with data_panel.container():
        with st.container(height=710, border=True):
            manifest_path = agent.output_dir / "manifest.json"
            report = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            rule_rows = report.get("rule_report", [])
            if rule_rows:
                if not st.session_state.output_current:
                    st.warning("Ниже отчет предыдущего запуска. После изменений выполните обработку заново.")
                st.markdown(f"**Что сделали при запуске · {report['as_of']} · версия: {(report.get('dataset_version') or {}).get('name', 'без имени')}**")
                for row in rule_rows:
                    status = "Применено" if row["status"] == "applied" else "Пропущено"
                    st.markdown(f"**{row['table']} · {row['rule_id']} v{row['version']} — {status}: {row['reason']}.** {row['result']}")
            raw_tab, output_tab = st.tabs(["Исходные", "Результаты"])
            with raw_tab:
                if not agent.tables:
                    st.info("Загрузите CSV или выберите пример сверху.")
                for name, df in agent.tables.items():
                    with st.expander(f"{name} · {len(df)} строк × {len(df.columns)} колонок", expanded=True):
                        st.dataframe(df.head(100), use_container_width=True, height=210)
                        st.caption(f"Показаны первые 100 строк. Пустых ячеек: {int(df.isna().sum().sum())} · полных дублей: {int(df.duplicated().sum())}")
            with output_tab:
                paths = sorted(agent.output_dir.glob("*.csv"))
                if paths and not rule_rows and not st.session_state.output_current:
                    st.warning("Это предыдущий результат. После новой загрузки или ошибки выполните обработку заново.")
                if not paths:
                    st.caption("После /run здесь появятся подготовленные таблицы и файлы для скачивания.")
                for path in paths:
                    df = pd.read_csv(path)
                    with st.expander(f"{path.stem} · {len(df)} строк", expanded=False):
                        st.dataframe(df.head(100), use_container_width=True)
                        st.download_button("Скачать CSV", path.read_bytes(), file_name=path.name, key="download_" + path.name)
                manifest = agent.root / "output" / "manifest.json"
                if manifest.exists():
                    st.download_button("Скачать manifest.json", manifest.read_bytes(), file_name="manifest.json")


render_logs()
render_memory()
render_data()
render_tokens()


def refresh(event):
    render_logs()
    render_tokens()
    if event["kind"] == "memory.write":
        st.session_state.output_current = False
        render_memory()


agent.on_event = refresh
if pending := st.session_state.pending:
    # The user message was rendered above, before any network request starts.
    try:
        with chat_box:
            with st.chat_message("assistant"):
                with st.spinner("Агент работает… Подробности в журнале слева."):
                    reply = agent.ask(pending, st.session_state.thread)
                st.write(reply)
        if any(e["kind"] == "tool.end" and e.get("name") == "run_transformations" for e in agent.events[-3:]):
            st.session_state.output_current = True
    except Exception as exc:
        reply = "Обработка остановлена: " + str(exc)
        st.session_state.output_current = False
        with chat_box:
            with st.chat_message("assistant"):
                st.error(reply)
    finally:
        agent.on_event = None
        st.session_state.pending = None
    st.session_state.chat.append(("assistant", reply))
    st.rerun()
agent.on_event = None
