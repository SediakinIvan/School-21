from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from typing import TypedDict, Optional
from typing_extensions import Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import json
from datetime import datetime, timedelta

API_KEY = "MDE5YTAyMjQtMmNkZC03NjI2LWFkMTQtMzEzNmZiYTRkZTQ2OmQ1ODUyOGQ0LTFhYjEtNDFkOC05MWI1LTY3YmNlNGU5ZmE0MQ=="


# Определяем состояние агента
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_action: Optional[str]
    result_data: Optional[dict]


# Инициализация модели
llm = GigaChat(credentials=API_KEY, verify_ssl_certs=False, model="GigaChat-2")


# Инструменты агента
@tool
def save_to_json(data: dict) -> str:  # Сохраняет данные в файл requests.json
    """Сохраняет данные в файл requests.json"""
    try:
        try:
            with open('requests.json', 'r', encoding='utf-8') as f:
                existing_data = json.load(f)  # чтение JSON данных из файла
        except FileNotFoundError:
            existing_data = []

        if not isinstance(existing_data, list):  # проверка чтобы в файле всегда был список
            existing_data = [existing_data]

        data['saved_at'] = datetime.now().isoformat()
        existing_data.append(data)  # добавили метку когда произведены сохранения

        with open('requests.json', 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=4, ensure_ascii=False)  # записали изменения в файлик

        return f"Сохранено. Всего записей: {len(existing_data)}"
    except Exception as e:
        return f"Ошибка: {str(e)}"


@tool
def read_from_json() -> list:  # Читает данные из файла requests.json
    """Читает данные из файла requests.json"""
    try:
        with open('requests.json', 'r', encoding='utf-8') as f:
            return json.load(f)  # вернули прочитанный файл
    except FileNotFoundError:
        return []


@tool
def filter_data(data: list, subject: str, days: int = 30) -> list:  # Фильтрует данные по предмету и дате
    """Фильтрует данные по предмету и дате"""
    filtered = []
    cutoff_date = datetime.now() - timedelta(days=days)

    for item in data:
        if item.get('subject') == subject:
            saved_at = item.get('saved_at', '')
            try:
                item_date = datetime.fromisoformat(saved_at)
                if item_date >= cutoff_date:
                    filtered.append(item)
            except ValueError:
                filtered.append(item)

    return filtered


tools = [save_to_json, read_from_json, filter_data]

# Системные промпты
CLASSIFY_PROMPT = SystemMessage(content="""
    Ты - помощник для классификации учебных материалов по предметам.
Определи, для какого из следующих предметов материал будет наиболее полезен:

1. Численные методы
2. Компьютерные сети
3. Программирование на python
4. Физика

Проанализируй содержание материала и верни ТОЛЬКО название подходящего предмета без дополнительных объяснений.
Если материал не подходит ни к одному предмету, верни "Другой предмет".

Список ссылок по предметам:
Численные методы
https://books.altspu.ru/document/65
https://openedu.ru/course/spbstu/NUMMETH/
http://wiki.cs.hse.ru/%D0%A7%D0%B8%D1%81%D0%BB%D0%B5%D0%BD%D0%BD%D1%8B%D0%B5_%D0%9C%D0%B5%D1%82%D0%BE%D0%B4%D1%8B_2021
https://www.hse.ru/edu/courses/339562855
https://teach-in.ru/course/numerical-methods-part-1
https://www.matburo.ru/st_subject.php?p=dr&rut=d992e77c9b77270bef82d706c585bfda4bdda23e35a9fb73a75809a9bc7c9608

Компьютерные сети
https://proglib.io/p/network-books
https://asozykin.ru/courses/networks_online
https://sites.google.com/view/malikov-m-v/%D1%81%D1%82%D1%83%D0%B4%D0%B5%D0%BD%D1%82%D0%B0%D0%BC/3-%D0%BA%D1%83%D1%80%D1%81/%D0%BA%D0%BE%D0%BC%D0%BF%D1%8C%D1%8E%D1%82%D0%B5%D1%80%D0%BD%D1%8B%D0%B5-%D1%81%D0%B5%D1%82%D0%B8
https://www.journal-altspu.ru/document/129
https://ru.hexlet.io/blog/posts/kompyuternaya-set-chto-eto-takoe-osnovnye-printsipy
https://gb.ru/courses/3731

Программирование на python
https://www.knorus.ru/catalog/informatika/698633-programmnaya-inzheneriya-bakalavriat-magistratura-uchebnik/
https://stepik.org/course/67/promo
https://ru.pythontutor.ru/problem/old/1
https://selectel.ru/blog/courses/course-python/
https://devpractice.ru/python/

Физика
https://madi.ru/438-kafedra-fizika-uchebnye-posobiya-po-lekcionnomu-kursu.html
https://znanierussia.ru/articles/%D0%9A%D0%BB%D0%B0%D1%81%D1%81%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B0%D1%8F_%D0%BC%D0%B5%D1%85%D0%B0%D0%BD%D0%B8%D0%BA%D0%B0
https://bigenc.ru/l/nachala-termodinamiki-7415b1
https://naked-science.ru/tags/elektrodinamika
https://nonfiction.ru/stream/kvantovaya-fizika-za-5-minut-glavnyie-voprosyi-i-idei

Если ссылки в этом списке нет, открой ее и проанализируй содержимое самостоятельно.

Верни ТОЛЬКО название предмета без лишних слов.
""")

REPORT_PROMPT = SystemMessage(content="""
Ты - помощник для генерации отчетов по учебным материалам.
Пользователь запрашивает отчет по определенному предмету за период времени.
Используй инструменты для чтения данных из файла и их фильтрации.
После получения данных сформируй отчет в виде структурированного списка.
""")

MAIN_PROMPT = SystemMessage(content="""
Ты помощник для учебных материалов. Определи что нужно:
- Ссылка → классифицируй
- Запрос отчета → сгенерируй отчет  
- Остальное → веди беседу
""")


# Узлы графа
def router(state: AgentState) -> AgentState:  # узел решает что делать с сообщениями пользователя
    last_msg = state["messages"][-1].content.lower()  # получает последнее соо пользователя

    if any(protocol in last_msg for protocol in ['http://', 'https://', 'www.']):
        return {"current_action": "classify"}
    elif any(keyword in last_msg for keyword in ['отчет', 'report', 'материалы', 'список']):
        return {"current_action": "report"}
    else:
        return {"current_action": "chat"}


def classify_node(state: AgentState) -> AgentState:  # кЛАССИФИКАТОР ССЫЛКИ
    user_msg = state["messages"][-1].content

    response = llm.invoke([CLASSIFY_PROMPT, HumanMessage(content=user_msg)])
    subject = response.content.strip()

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "subject": subject,
        "original_link": user_msg
    }

    save_result = save_to_json.invoke({"data": result})

    return {
        "result_data": result,
        "messages": [HumanMessage(content=f"Классифицировано: {subject}. {save_result}")]
    }


def report_node(state: AgentState) -> AgentState:  # создает отчет, по умолчанию предмет - прога на питоне
    user_query = state["messages"][-1].content

    # Определяем предмет
    subjects = ["Численные методы", "Компьютерные сети", "Программирование на python", "Физика"]
    subject = None

    for subj in subjects:
        if subj.lower() in user_query.lower():
            subject = subj
            break

    if not subject:
        # Если предмет не найден, используем LLM для определения
        subject_prompt = f"""
        Пользователь запрашивает: {user_query}

        Определи, для какого предмета нужен отчет из списка:
        - Численные методы
        - Компьютерные сети
        - Программирование на python  
        - Физика

        Верни ТОЛЬКО название предмета.
        """
        subject_response = llm.invoke([HumanMessage(content=subject_prompt)])
        subject = subject_response.content.strip()

    # Определяем период
    days = 30
    if "недел" in user_query:
        days = 7
    elif "месяц" in user_query:
        days = 30
    elif "квартал" in user_query:
        days = 90
    elif "год" in user_query:
        days = 365

    # Получаем данные
    all_data = read_from_json.invoke({})
    filtered_data = filter_data.invoke({"data": all_data, "subject": subject, "days": days})

    return {
        "result_data": filtered_data,
        "messages": [HumanMessage(content=f"Отчет: {subject} за {days} дней - {len(filtered_data)} записей")]
    }


def chat_node(state: AgentState) -> AgentState:  # просто общение с пользователем
    # Используем обычный llm без инструментов для избежания ошибок
    response = llm.invoke([MAIN_PROMPT] + state["messages"])
    return {"messages": [response]}


def output_node(state: AgentState) -> AgentState:  # ответ пользователю
    """Формирует ответ"""
    if state.get("result_data"):  # проверка есть ли вообще какие-то данные для вывода
        result = state["result_data"]
        return {"messages": [HumanMessage(content=json.dumps(result, ensure_ascii=False, indent=2))]}
    return state


# Создаем агента
def create_agent():
    workflow = StateGraph(AgentState)  # создали пустую карту

    workflow.add_node("router", router)
    workflow.add_node("classify", classify_node)
    workflow.add_node("report", report_node)
    workflow.add_node("chat", chat_node)
    workflow.add_node("output", output_node)

    workflow.set_entry_point("router")  # начинать всегда с распределителя

    workflow.add_conditional_edges(  # добавление условных ребер, развилка задач
        "router",  # начало
        lambda state: state.get("current_action", "chat"),  # решаем куда идти
        {"classify": "classify", "report": "report", "chat": "chat"}
    )
    # прямые указатели после чего куда идти
    workflow.add_edge("classify", "output")
    workflow.add_edge("report", "output")
    workflow.add_edge("chat", "output")
    workflow.add_edge("output", END)

    return workflow.compile()


# Запуск агента
agent = create_agent()


def process_input(user_input: str):  # обрабатывает запрос пользователя
    print(f"\nтвоё соо: {user_input}")

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)],
        "current_action": None,
        "result_data": None
    })

    response = result["messages"][-1].content
    print(f"соо агента: {response}")


if __name__ == "__main__":
    print(" Реактивный ИИ-агент для учебных материалов")
    print("=" * 50)
    print("Функциональность:")
    print("- Классификация ссылок по предметам")
    print("- Генерация отчетов по предметам и периодам")
    print("- Сохранение данных в JSON формате")
    print("=" * 50)
    print("Примеры команд:")
    print("- https://example.com (классификация ссылки)")
    print("- отчет по физике за неделю")
    print("- материалы по python за месяц")
    print("- выход (для завершения)")
    print("=" * 50)

    while True:
        user_input = input("\nвведите соо  ").strip()

        if user_input.lower() in ['exit', 'выход']:
            break

        if user_input:
            try:
                process_input(user_input)
            except Exception as e:
                print(f"Ошибка: {e}")